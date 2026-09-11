"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import get_context
from multiprocessing.connection import wait
import time
from abc import ABC, abstractmethod
import copy


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents":
            remote.send((env.n_agents))
        elif cmd == "call":
            method_name, args, kwargs = data
            result = getattr(env, method_name)(*args, **kwargs)
            remote.send(result)
        else:
            raise NotImplementedError


class WorkerError(RuntimeError):
    """An environment process exited before completing its request."""


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None, *, response_timeout=30.0,
                 startup_timeout=120.0, close_timeout=2.0):
        if not env_fns or min(response_timeout, startup_timeout, close_timeout) <= 0:
            raise ValueError("Require environments and positive timeouts")
        self.waiting = False
        self.closed = False
        self.response_timeout = response_timeout
        self.close_timeout = close_timeout
        self.remotes, self.ps = [], []
        # Explicit spawn prevents siblings from retaining each other's pipe ends.
        context = get_context("spawn")
        try:
            for env_fn in env_fns:
                remote, child_remote = context.Pipe()
                self.remotes.append(remote)
                process = context.Process(target=shareworker,
                    args=(child_remote, CloudpickleWrapper(env_fn)), daemon=True)
                try:
                    process.start()
                    self.ps.append(process)
                finally:
                    child_remote.close()
            self._send(0, "get_num_agents", None)
            self.n_agents = self._receive([0], startup_timeout)[0]
            self._send(0, "get_spaces", None)
            observation_space, share_observation_space, action_space = self._receive([0], startup_timeout)[0]
            super().__init__(len(env_fns), observation_space, share_observation_space, action_space)
        except BaseException:
            self.close(force=True)
            raise

    def _idle(self):
        if self.closed or self.waiting:
            raise RuntimeError("Environment is closed or a step is already pending")

    def _send(self, index, command, data):
        try:
            self.remotes[index].send((command, data))
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close(force=True)
            raise WorkerError(f"Cannot send {command} to worker {index}") from exc

    def _receive(self, indices, timeout=None):
        indices = list(indices)
        pending = set(indices)
        results = {}
        deadline = time.monotonic() + (self.response_timeout if timeout is None else timeout)
        try:
            while pending:
                dead = [dict(index=i, pid=self.ps[i].pid, exitcode=self.ps[i].exitcode)
                        for i in pending if self.ps[i].exitcode is not None]
                if dead:
                    raise WorkerError(f"Environment worker exited: {dead}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Environment response timed out; workers={sorted(pending)}")
                ready = wait([self.remotes[i] for i in pending] +
                             [self.ps[i].sentinel for i in pending], timeout=remaining)
                for i in sorted(pending):
                    if self.ps[i].sentinel in ready:
                        self.ps[i].join(timeout=0)
                        raise WorkerError(f"Environment worker {i}, pid={self.ps[i].pid}, "
                                          f"exited with code {self.ps[i].exitcode}")
                    if self.remotes[i] in ready:
                        try:
                            results[i] = self.remotes[i].recv()
                        except (EOFError, OSError) as exc:
                            raise WorkerError(f"Environment worker {i}, pid={self.ps[i].pid}, "
                                              "closed its response pipe") from exc
                        pending.remove(i)
            return [results[i] for i in indices]
        except BaseException:
            self.close(force=True)
            raise

    def step_async(self, actions):
        self._idle()
        if len(actions) != len(self.remotes):
            raise ValueError("Require one action batch per environment")
        self.waiting = True
        for i, action in enumerate(actions):
            self._send(i, "step", action)

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("No environment step is pending")
        try:
            results = self._receive(range(len(self.remotes)))
        finally:
            self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (np.stack(obs), np.stack(share_obs), np.stack(rews),
                np.stack(dones), infos, np.stack(available_actions))

    def reset(self):
        self._idle()
        for i in range(len(self.remotes)):
            self._send(i, "reset", None)
        results = self._receive(range(len(self.remotes)))
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        self._idle()
        for i in range(len(self.remotes)):
            self._send(i, "reset_task", None)
        return np.stack(self._receive(range(len(self.remotes))))

    def close(self, *, force=False):
        if self.closed:
            return
        self.closed = True
        force = force or self.waiting or any(not p.is_alive() for p in self.ps)
        self.waiting = False
        if not force:
            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except (EOFError, OSError):
                    pass
        else:
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
        for remote in self.remotes:
            remote.close()
        # A single shared deadline bounds each phase, independent of worker count.
        for action in (None, "terminate", "kill"):
            alive = [p for p in self.ps if p.is_alive()]
            if action is not None:
                for p in alive:
                    getattr(p, action)()
            deadline = time.monotonic() + self.close_timeout
            for p in self.ps:
                p.join(timeout=max(0, deadline-time.monotonic()))

    def call_at(self, index, method_name, *args, **kwargs):
        self._idle()
        index = int(index)
        self._send(index, "call", (method_name, args, kwargs))
        return self._receive([index])[0]


# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def call_at(self, index, method_name, *args, **kwargs):
        env = self.envs[int(index)]
        return getattr(env, method_name)(*args, **kwargs)
