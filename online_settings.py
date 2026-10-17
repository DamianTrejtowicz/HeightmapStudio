"""English account settings for optional online AI services."""
import copy
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from online_ai import PROVIDERS, OnlineEngine, validate


class SettingsDialog:
    def __init__(self, parent, config, store, on_save):
        self.config, self.store, self.on_save = copy.deepcopy(config), store, on_save
        self.events = queue.Queue()
        self.window = tk.Toplevel(parent)
        self.window.title('AI service settings — Heightmap Studio')
        scale=max(1,float(self.window.tk.call('tk','scaling'))/(96/72))
        width=min(self.window.winfo_screenwidth()-70,round(760*scale))
        height=min(self.window.winfo_screenheight()-90,round(670*scale))
        self.window.geometry(f'{width}x{height}'); self.window.minsize(min(700,width),min(580,height))
        self.window.transient(parent)
        self.window.grab_set()
        # Keep Save/Cancel visible even on small screens or with Windows text scaling.
        buttons=ttk.Frame(self.window,padding=(20,8,20,12)); buttons.pack(side='bottom',fill='x')
        viewport=ttk.Frame(self.window); viewport.pack(fill='both',expand=True)
        canvas=tk.Canvas(viewport,highlightthickness=0,bg='#edf1f7')
        scrollbar=ttk.Scrollbar(viewport,orient='vertical',command=canvas.yview)
        scrollbar.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        frame=ttk.Frame(canvas,padding=20)
        item=canvas.create_window((0,0),window=frame,anchor='nw')
        frame.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(item,width=e.width))
        self.window.bind('<MouseWheel>',lambda e:canvas.yview_scroll(-int(e.delta/120),'units'))
        ttk.Label(frame,text='AI services & accounts',font=('Segoe UI Semibold',19)).pack(anchor='w')
        ttk.Label(frame,text='Offline always works without a key. Online uploads require clicking Generate.',wraplength=690).pack(anchor='w',pady=(7,14))
        self.values={}
        self.keys={}
        row=ttk.Frame(frame); row.pack(fill='x')
        ttk.Label(row,text='Selected service:').pack(side='left',padx=(0,10))
        self.values['provider']=tk.StringVar(value=config.provider)
        ttk.Combobox(row,textvariable=self.values['provider'],values=PROVIDERS,state='readonly',width=39).pack(side='left',fill='x',expand=True)
        tabs=ttk.Notebook(frame); tabs.pack(fill='both',expand=True,pady=16)
        for service,title in [('hf','Hugging Face'),('openai','OpenAI'),('gemini','Google Gemini')]:
            tab=ttk.Frame(tabs,padding=16); tabs.add(tab,text=title)
            if service=='hf':
                text='Depth estimation on Hugging Face Spaces. The default Space has a free quota; availability and queue limits apply. An optional HF token uses your account quota. Compatible Spaces must expose /on_submit and a grayscale map as the second output.'
                field,title2,choices='hf_space','Space (owner/name):',None
            elif service=='openai':
                text='Experimental artistic relief using the Images API. Requires your own API key and image-model access. API usage can incur charges. A ChatGPT subscription is not an API key. Generated shapes may differ from the original; inspect alignment before printing.'
                field,title2,choices='openai_model','Image model:',['gpt-image-2.5-sunburst','gpt-image-2.5-flare','gpt-image-2','gpt-image-1.5']
            else:
                text='Experimental artistic relief using the Gemini Interactions API. Requires your own AI Studio API key and an image-generation model. Quotas, model availability and charges depend on your account. Generated details may differ from the source.'
                field,title2,choices='gemini_model','Image model:',['gemini-nano-banana-2.1','gemini-3.1-flash-lite-image','gemini-3.1-flash-image','gemini-3-pro-image']
            ttk.Label(tab,text=text,wraplength=650,justify='left').pack(anchor='w',pady=(0,20))
            ttk.Label(tab,text=title2).pack(anchor='w')
            self.values[field]=tk.StringVar(value=getattr(config,field))
            widget=ttk.Combobox(tab,textvariable=self.values[field],values=choices) if choices else ttk.Entry(tab,textvariable=self.values[field])
            widget.pack(fill='x',pady=(4,18))
            ttk.Label(tab,text='HF access token (optional):' if service=='hf' else 'API key:').pack(anchor='w')
            self.keys[service]=tk.StringVar(value=config.keys.get(service,''))
            keyrow=ttk.Frame(tab); keyrow.pack(fill='x',pady=4)
            entry=ttk.Entry(keyrow,textvariable=self.keys[service],show='•'); entry.pack(side='left',fill='x',expand=True)
            visible=tk.BooleanVar(value=False)
            ttk.Checkbutton(keyrow,text='Show',variable=visible,
                command=lambda e=entry,v=visible:e.configure(show='' if v.get() else '•')).pack(side='left',padx=(8,0))
        bottom=ttk.Frame(frame); bottom.pack(fill='x')
        ttk.Label(bottom,text='Request timeout:').pack(side='left')
        self.values['timeout']=tk.StringVar(value=str(config.timeout))
        ttk.Combobox(bottom,textvariable=self.values['timeout'],values=['60','120','240','600'],width=7).pack(side='left',padx=8)
        ttk.Label(bottom,text='seconds').pack(side='left')
        self.remember=tk.BooleanVar(value=config.remember)
        ttk.Checkbutton(frame,text='Remember keys on this PC (encrypted with Windows DPAPI)',variable=self.remember).pack(anchor='w',pady=(12,4))
        ttk.Label(frame,text='Unchecked: keys stay in memory until the app closes. No passwords are requested.',wraplength=690).pack(anchor='w')
        self.result=tk.StringVar(value='Test connection checks account/Space access; it does not generate or upload images.')
        ttk.Label(frame,textvariable=self.result,wraplength=690).pack(anchor='w',pady=12)
        self.test_button=ttk.Button(buttons,text='Test connection',command=self.test); self.test_button.pack(side='left')
        ttk.Button(buttons,text='Forget saved keys',command=self.forget).pack(side='left',padx=8)
        ttk.Button(buttons,text='Cancel',command=self.window.destroy).pack(side='right')
        ttk.Button(buttons,text='Save settings',command=self.save,style='Accent.TButton').pack(side='right',padx=8)
        self.window.after(100,self.poll)

    def draft(self):
        config=copy.deepcopy(self.config)
        for name,var in self.values.items(): setattr(config,name,var.get().strip())
        config.timeout=int(config.timeout)
        config.keys={name:var.get().strip() for name,var in self.keys.items()}
        config.remember=self.remember.get()
        validate(config)
        return config

    def save(self):
        try:
            config=self.draft()
            self.store.save(config)
            self.on_save(config)
            self.window.destroy()
        except Exception as exc: messagebox.showerror('Settings',str(exc),parent=self.window)

    def forget(self):
        try:
            config=self.draft(); config.keys={}; config.remember=False
            self.store.save(config)
            for var in self.keys.values(): var.set('')
            self.remember.set(False)
            self.on_save(config)
            self.result.set('Saved and in-memory keys have been cleared.')
        except Exception as exc: messagebox.showerror('Settings',str(exc),parent=self.window)

    def test(self):
        try: config=self.draft()
        except Exception as exc:
            self.result.set(str(exc)); return
        self.test_button.configure(state='disabled'); self.result.set('Checking selected service…')
        def run():
            try: self.events.put(OnlineEngine().check(config))
            except Exception as exc: self.events.put(str(exc))
        threading.Thread(target=run,daemon=True).start()

    def poll(self):
        if not self.window.winfo_exists(): return
        try:
            self.result.set(self.events.get_nowait())
            self.test_button.configure(state='normal')
        except queue.Empty: pass
        self.window.after(100,self.poll)
