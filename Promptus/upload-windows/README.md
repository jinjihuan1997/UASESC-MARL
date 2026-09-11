该文件夹下的内容主要负责将视频数据传输给地面幸存者
2种传输方式
1: 将视频发布到网页上, 用户自主访问内容
2: 将视频点对点传输给用户, 必须要考虑对接软件

代码文件: webrtc_dirstream_server.py, 负责将帧文件整合为视频推流到网页端, 多路上传
运行指令 -- 推流:
pip install aiortc aiohttp av opencv-python
python upload-windows\webrtc_dirstream_server.py --host 0.0.0.0 --port 8080 `
  --add "name=ocean;src=C:\Users\ENeS\Desktop\GAI-SC\Promptus\data\ocean\results\rank8_interval10;fps=15;w=512;h=512;fit=pad" `
  --add "name=sky;src=C:\Users\ENeS\Desktop\GAI-SC\Promptus\data\sky\results\rank8_interval10;fps=15;w=512;h=512;fit=pad" `
  --add "name=uvg;src=C:\Users\ENeS\Desktop\GAI-SC\Promptus\data\uvg\results\rank8_interval10;fps=15;w=512;h=512;fit=pad"
