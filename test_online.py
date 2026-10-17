from pathlib import Path
from unittest.mock import patch, Mock
import base64
import io
import json
import tempfile
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageGrab
import online_ai as net
import heightmap_studio as app
from online_settings import SettingsDialog

work=Path(__file__).resolve().parent/'.test-output'/'online'
work.mkdir(parents=True,exist_ok=True)
source=Image.new('RGBA',(200,100),(90,150,210,255))
output=Image.linear_gradient('L').resize((1024,1024))
buffer=io.BytesIO(); output.save(buffer,format='PNG')
encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
dummy='test-only-sentinel-not-a-real-API-key'

with tempfile.TemporaryDirectory() as folder:
    store=net.ConfigStore(folder)
    config=net.OnlineConfig(remember=True,keys={'openai':dummy,'hf':dummy})
    store.save(config)
    assert dummy not in store.path.read_text()
    assert store.load().keys==config.keys
    config.remember=False; store.save(config)
    assert store.load().keys=={} and 'protected_keys' not in store.path.read_text()

for provider,key,reply in [
    (net.PROVIDERS[2],'openai',{'data':[{'b64_json':encoded}]}),
    (net.PROVIDERS[3],'gemini',{'status':'completed','steps':[{'type':'model_output','content':[{'type':'image','data':encoded}]}]}),
]:
    config=net.OnlineConfig(provider=provider,keys={key:dummy})
    with patch('online_ai.request_json',return_value=reply) as request:
        raw=net.OnlineEngine().estimate(source,config)
        assert raw.shape==(100,200) and np.isfinite(raw).all()
        args,kwargs=request.call_args
        assert args[0]=='POST'
        if key=='openai':
            assert args[1]=='https://api.openai.com/v1/images/edits'
            assert kwargs['data']['model']==config.openai_model
            assert kwargs['files']['image[]'][2]=='image/png'
            assert kwargs['headers']['Authorization']=='Bearer '+dummy
        else:
            assert args[1]=='https://generativelanguage.googleapis.com/v1beta/interactions'
            assert kwargs['json']['store'] is False
            assert kwargs['json']['response_format']['type']=='image'
            assert kwargs['headers']['x-goog-api-key']==dummy
    with patch('online_ai.request_json',side_effect=RuntimeError('Denied '+dummy)):
        try: net.OnlineEngine().estimate(source,config)
        except RuntimeError as e: assert dummy not in str(e)
        else: raise AssertionError('Expected a safe service error')
    config.keys={}
    with patch('online_ai.request_json') as request:
        try: net.OnlineEngine().estimate(source,config)
        except RuntimeError: pass
        else: raise AssertionError('Missing key must block generation')
        request.assert_not_called()

response=Mock(status_code=429)
response.json.return_value={'error':{'message':'Quota limit'}}
with patch('requests.request',return_value=response):
    try: net.request_json('GET','https://api.openai.com/v1/models',30)
    except RuntimeError as e: assert 'Quota' in str(e) and '429' in str(e)

with tempfile.TemporaryDirectory() as folder:
    store=net.ConfigStore(folder)
    with patch('heightmap_studio.ConfigStore',return_value=store):
        root=tk.Tk(); ui=app.Application(root)
    path=work/'synthetic.png'; source.save(path)
    ui.online_engine.estimate=Mock(return_value=np.asarray(Image.linear_gradient('L').resize(source.size),dtype=np.float32))
    ui.provider_var.set(net.PROVIDERS[1]); ui.provider_changed()
    with patch('tkinter.filedialog.askopenfilename',return_value=str(path)):
        ui.load()
    started=time.monotonic()
    while ui.busy:
        root.update(); time.sleep(.02)
        if time.monotonic()-started>15: raise TimeoutError('Load timed out')
    assert ui.raw is None
    ui.online_engine.estimate.assert_not_called()
    ui.generate()
    while ui.busy: root.update(); time.sleep(.02)
    ui.online_engine.estimate.assert_called_once()
    assert ui.raw is not None
    config=net.OnlineConfig(provider=net.PROVIDERS[1])
    dialog=SettingsDialog(root,config,store,ui.settings_saved)
    root.update()
    assert all(var.get()=='' for var in dialog.keys.values())
    dialog.window.after(250,lambda:None)
    root.update()
    import ctypes
    hwnd=ctypes.windll.user32.GetParent(dialog.window.winfo_id())
    ImageGrab.grab(window=hwnd).save(work/'settings_preview.png')
    dialog.keys['openai'].set(dummy)
    dialog.remember.set(True)
    dialog.save()
    assert store.load().keys['openai']==dummy
    assert ui.online_config.keys['openai']==dummy
    assert dummy not in store.path.read_text()
    root.destroy()

report={'version':app.VERSION,'encrypted_credentials_roundtrip':'passed','session_only_keys_not_saved':'passed',
        'OpenAI_and_Gemini_request_and_response_contracts':'passed (mock HTTP; no live paid calls)',
        'missing_key_blocks_network':'passed','error_secret_redaction':'passed',
        'quota_error_feedback':'passed','no_upload_on_load_or_service_selection':'passed',
        'online_Generate_action_dispatch':'passed','English_settings_UI_and_save':'passed'}
(work/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
