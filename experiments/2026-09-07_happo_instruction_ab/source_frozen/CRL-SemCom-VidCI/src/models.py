import torch
import torch.nn as nn
import torch.nn.functional as F
import shutters.shutters_adaptive5_nomask as shutters

from experiment_scripts.comm.transmission import SemanticTransmissionSystem

device = 'cuda:0'
from nets.MST_Plus_Plus_nomask import MST_Plus_Plus


def define_shutter(shutter_type, args, test=False, model_dir=''):
    return shutters.Shutter(shutter_type=shutter_type, block_size=args.block_size,
                            test=test, resume=args.resume, model_dir=model_dir, init=args.init,
                            legacy_action_space=getattr(args, 'legacy_action_space', False))


def define_model(shutter, decoder, args, get_coded=False):
    ''' Define any special interpolation modules in between encoder and decoder '''
    if args.interp not in [None, 'none', 'scatter']:
        raise NotImplementedError(f'Interpolation mode {args.interp} is not implemented')

    transmission = define_transmission(args)
    return Model(
        shutter,
        decoder,
        transmission=transmission,
        dec_name=args.decoder,
        get_coded=get_coded,
        shutter_loss_weight=getattr(args, 'shutter_loss_weight', 1.0),
        comm_cost_weight=getattr(args, 'mu_comm', 1e-3),
        shutter_cost_weight=getattr(args, 'sci_cost_weight', 0.05),
    )



def define_decoder(model_name, args):
    if args.decoder == 'none':
        return None
    out_ch = 16
    if args.shutter == 'full':
        in_ch = 3
    elif args.shutter in ['short', 'med', 'long'] or args.interp is None:
        in_ch = 1
    elif 'quad' in args.shutter:
        in_ch = 4
    elif 'nonad' in args.shutter:
        in_ch = 9
    elif args.interp == 'scatter':
        in_ch = 9
    else:
        raise NotImplementedError

    if model_name == 'MST':
        model = MST_Plus_Plus()
        return model

    raise NotImplementedError('Model not specified correctly')


def define_transmission(args):
    if not getattr(args, 'use_semantic_comm', False):
        return None

    return SemanticTransmissionSystem(
        sensor_channels=getattr(args, 'comm_sensor_channels', 24),
        latent_channels=getattr(args, 'comm_latent_channels', 48),
        hidden_channels=getattr(args, 'comm_hidden_channels', 64),
        rate_levels=getattr(args, 'comm_rate_levels', 4),
        snr_db=getattr(args, 'comm_snr_db', 10.0),
        forced_rate_level=getattr(args, 'comm_forced_rate_level', None),
        channel_coding_rate=getattr(args, 'comm_channel_coding_rate', 0.5),
        modulation_order=getattr(args, 'comm_modulation_order', 4),
    )



class Model(nn.Module):
    def __init__(
        self,
        shutter,
        decoder,
        transmission=None,
        dec_name=None,
        get_coded=False,
        shutter_loss_weight=1.0,
        comm_cost_weight=1e-3,
        shutter_cost_weight=0.05,
    ):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder
        self.transmission = transmission

        self.dec_name = dec_name
        self.shutter_loss_weight = shutter_loss_weight
        self.comm_cost_weight = comm_cost_weight
        self.shutter_cost_weight = shutter_cost_weight

    @staticmethod
    def _build_shutter_cost(actions):
        actions = actions.float()
        costs = torch.zeros_like(actions)
        costs = torch.where(actions == 0, torch.ones_like(costs), costs)
        costs = torch.where(actions == 1, 2 * torch.ones_like(costs), costs)
        costs = torch.where(actions == 2, 4 * torch.ones_like(costs), costs)
        costs = torch.where(actions == 3, 8 * torch.ones_like(costs), costs)
        return costs

    def forward(self, input, train=True, steps=None, forced_rate_level=None, forced_ratio_level=None):
        shutter_train = train and any(param.requires_grad for param in self.shutter.parameters())
        coded1, actions, rate_loss = self.shutter(
            input, train=shutter_train, steps=steps, forced_ratio_level=forced_ratio_level)
        if not coded1.requires_grad:
            ## needed for computing gradients wrt input for fixed shutters
            coded1.requires_grad = True

        if self.decoder is None:
            if self.get_coded:
                return coded1, coded1
            return coded1

        comm_out = None
        coded_for_decoder = coded1
        if self.transmission is not None:
            coded_for_decoder, comm_out = self.transmission(
                coded1,
                train=train,
                forced_rate_level=forced_rate_level,
            )
            shutter_cost = self._build_shutter_cost(actions)
            pooled_shape = comm_out['rate_level_map'].shape[-2:]
            pooled_shutter_cost = F.adaptive_avg_pool2d(shutter_cost, pooled_shape)
            shutter_rate_scale = self.shutter_cost_weight / max(self.comm_cost_weight, 1e-8)
            comm_out['reward_rate_map'] = (
                comm_out['reward_rate_map'] + shutter_rate_scale * pooled_shutter_cost.detach()
            )

            if rate_loss is not None:
                pooled_shutter_log_prob = F.adaptive_avg_pool2d(rate_loss, pooled_shape)
                if comm_out['log_prob'] is None:
                    comm_out['log_prob'] = self.shutter_loss_weight * pooled_shutter_log_prob
                else:
                    comm_out['log_prob'] = comm_out['log_prob'] + self.shutter_loss_weight * pooled_shutter_log_prob
                comm_out['shutter_log_prob'] = rate_loss
                comm_out['shutter_cost_map'] = pooled_shutter_cost
        # self.decoder.eval()
        # x = self.decoder(torch.concat([coded1,actions],dim=1))
        x = self.decoder(coded_for_decoder, actions)
        # x=coded1[:,:-1,:,:]

        if self.get_coded:
            return x, coded_for_decoder
        if comm_out is not None:
            return x, coded_for_decoder, actions, {
                'shutter_log_prob': rate_loss,
                'comm': comm_out,
            }
        return x, coded1, actions, rate_loss

    def forward_using_capture(self, coded):
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x
