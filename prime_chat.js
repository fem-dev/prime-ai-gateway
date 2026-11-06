function render({ model, el }) {
  el.classList.add("prime-chat-widget");
  el.innerHTML = '<section class="pc-shell" dir="rtl" lang="fa"><header class="pc-top"><div class="pc-brand"><b>✦</b><span><strong>Prime AI</strong><small>گفت‌وگوی هوشمند</small></span></div><div class="pc-tools"><span class="pc-status">● <i>آماده</i></span><button class="pc-new">＋ گفت‌وگوی جدید</button></div></header><main class="pc-scroll"><div class="pc-welcome"><div>◉</div><h2>از کجا شروع کنیم؟</h2><p>هر چیزی می‌خواهی بپرس یا یک فایل برای بررسی بفرست.</p><nav><button data-prompt="سلام! خودت را معرفی کن.">معرفی Prime AI</button><button data-prompt="یک ایده برای پروژه برنامه‌نویسی بده.">ایده برنامه‌نویسی</button><button data-prompt="یک تصویر فانتزی از شهری روی ابرها بساز.">ساخت تصویر</button></nav></div><div class="pc-messages"></div></main><footer><div class="pc-files"></div><div class="pc-compose"><input type="file" class="pc-file" multiple hidden accept="image/*,.pdf,.txt,.md,.csv,.json,.py,.html,.xml,.yaml,.yml"><button class="pc-add" title="افزودن فایل">＋</button><textarea rows="1" placeholder="پیامت را بنویس..."></textarea><button class="pc-stop" title="توقف" hidden>■</button><button class="pc-send" title="ارسال">➤</button></div><small>Prime AI ممکن است اشتباه کند؛ پاسخ‌های مهم را بررسی کن.</small></footer></section>';
  const q=s=>el.querySelector(s), area=q(".pc-scroll"), feed=q(".pc-messages"), input=q("textarea"), file=q(".pc-file");
  let history=[], uploads=[], busy=false, rid=0;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function md(s) {
    let t=esc(s||"");
    t=t.replace(/```([\\w+-]*)\\n([\\s\\S]*?)```/g,(_,lang,code)=>'<pre><label>'+esc(lang||"code")+'</label><button class="pc-copy">کپی</button><code>'+code+'</code></pre>');
    t=t.replace(/\x60([^\x60]+)\x60/g,"<code>$1</code>").replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/^### (.+)$/gm,"<h3>$1</h3>").replace(/^## (.+)$/gm,"<h2>$1</h2>").replace(/^&gt; (.+)$/gm,"<blockquote>$1</blockquote>").replace(/^[-*] (.+)$/gm,"<li>$1</li>").replace(/\\n/g,"<br>");
    return t;
  }
  function draw() {
    q(".pc-welcome").hidden=history.length>0;
    feed.innerHTML=history.map(m=>'<article class="pc-msg '+m.role+'"><span class="pc-avatar">'+(m.role==="user"?"ش":"✦")+'</span><div class="pc-body">'+md(m.content)+(m.image_data?'<img alt="تصویر تولیدشده" src="'+m.image_data+'"><a class="pc-download" download="prime-ai-image.png" href="'+m.image_data+'">دانلود تصویر</a>':"")+(m.choice?'<div class="pc-model-choices"><button data-model="qwen">Qwen-Image <small>مدل تصویرساز Qwen</small></button><button data-model="flux">FLUX.1-dev <small>مدل تصویرساز Flux</small></button></div>':"")+(m.files&&m.files.length?'<div class="pc-filechips">'+m.files.map(n=>'<span>📎 '+esc(n)+'</span>').join("")+'</div>':"")+'</div></article>').join("");
    feed.querySelectorAll(".pc-copy").forEach(b=>b.onclick=()=>{navigator.clipboard?.writeText(b.parentElement.querySelector("code").textContent);b.textContent="کپی شد";});
    feed.querySelectorAll("[data-model]").forEach(b=>b.onclick=()=>chooseImageModel(b.dataset.model,history.slice().reverse().find(m=>m.choice)?.prompt));
    area.scrollTop=area.scrollHeight;
  }
  async function chooseImageModel(choice,prompt) {
    if(!busy||!prompt)return;
    let hfToken="";
    if(choice==="flux") {
      const overlay=document.createElement("div");
      overlay.className="pc-token-overlay";
      overlay.innerHTML='<form class="pc-token-dialog" dir="rtl"><h3>توکن Hugging Face</h3><p>برای بارگذاری FLUX، توکن معتبر حسابی را وارد کن که دسترسی مدل FLUX.1-dev را دارد. توکن در فایل یا GitHub ذخیره نمی‌شود.</p><input type="password" autocomplete="off" spellcheck="false" placeholder="hf_..." aria-label="Hugging Face token" required><div><button type="button" class="pc-token-cancel">انصراف</button><button type="submit">ادامه</button></div><small>توکن فقط برای همین درخواست به هستهٔ برنامه فرستاده می‌شود.</small></form>';
      el.querySelector(".pc-shell").appendChild(overlay);
      const input=overlay.querySelector("input");
      input.focus();
      hfToken=await new Promise(resolve=>{
        overlay.querySelector("form").onsubmit=event=>{event.preventDefault();const value=input.value.trim();overlay.remove();resolve(value);};
        overlay.querySelector(".pc-token-cancel").onclick=()=>{overlay.remove();resolve("");};
      });
      if(!hfToken){busy=false;status("درخواست لغو شد",false);return;}
    }
    history=history.filter(m=>!m.choice);
    draw();
    status(choice==="flux"?"در حال آماده‌سازی FLUX.1-dev...":"در حال آماده‌سازی Qwen-Image...",true);
    model.set("request",JSON.stringify({id:rid,text:"",history:history.map(m=>({role:m.role,content:m.content})),attachments:[],image_prompt:prompt,image_model:choice,hf_token:hfToken}));
    model.save_changes();
  }
  function status(s,thinking) {
    q(".pc-status i").textContent=s;
    q(".pc-status").classList.toggle("thinking",!!thinking);
    q(".pc-send").classList.toggle("thinking",!!thinking);
    q(".pc-send").hidden=!!thinking;
    q(".pc-stop").hidden=!thinking;
  }
  function drawFiles() {
    q(".pc-files").innerHTML=uploads.map((f,i)=>'<span>📎 '+esc(f.name)+' <button data-i="'+i+'">×</button></span>').join("");
    q(".pc-files").querySelectorAll("button").forEach(b=>b.onclick=()=>{uploads.splice(+b.dataset.i,1);drawFiles();});
  }
  function addFiles(list) {
    Array.from(list).forEach(f=>{
      if(f.size>5*1024*1024){status("حداکثر ۵ مگابایت",false);return;}
      const r=new FileReader();
      r.onload=()=>{uploads.push({name:f.name,type:f.type||"application/octet-stream",data:String(r.result).split(",")[1]});drawFiles();};
      r.readAsDataURL(f);
    });
  }
  function send() {
    const text=input.value.trim();
    if(busy||(!text&&!uploads.length))return;
    const files=uploads.splice(0);drawFiles();
    history.push({role:"user",content:text,files:files.map(f=>f.name)});
    draw();input.value="";input.style.height="auto";busy=true;status("در حال فکر کردن...",true);
    rid++;
    model.set("request",JSON.stringify({id:rid,text:text,history:history.slice(0,-1).map(m=>({role:m.role,content:m.content})),attachments:files}));
    model.save_changes();
  }
  q(".pc-send").onclick=send;
  q(".pc-stop").onclick=()=>{if(!busy)return;model.set("cancel_request",String(rid));model.save_changes();busy=false;status("درخواست لغو شد",false);history.push({role:"assistant",content:"درخواست متوقف شد."});draw();};
  input.onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}};
  input.oninput=()=>{input.style.height="auto";input.style.height=Math.min(input.scrollHeight,140)+"px";};
  q(".pc-add").onclick=()=>file.click();
  file.onchange=()=>{addFiles(file.files);file.value="";};
  q(".pc-new").onclick=()=>{if(busy)return;history=[];feed.innerHTML="";q(".pc-welcome").hidden=false;status("آماده",false);};
  el.querySelectorAll("[data-prompt]").forEach(b=>b.onclick=()=>{input.value=b.dataset.prompt;send();});
  model.on("change:response",()=>{
    let d={};try{d=JSON.parse(model.get("response")||"{}");}catch(e){}
    if(d.id!==rid||!busy)return;
    if(d.error)history.push({role:"assistant",content:d.error});
    else if(d.choose_image_model)history.push({role:"assistant",content:d.text||"مدل تصویرساز را انتخاب کن:",choice:true,prompt:d.prompt||""});
    else if(d.text)history.push({role:"assistant",content:d.text,image_data:d.image_data||null});
    draw();busy=!!d.choose_image_model;status(d.choose_image_model?"انتخاب مدل تصویر":"آماده",!!d.choose_image_model);
  });
  model.on("change:status",()=>{if(busy)status(model.get("status")||"در حال پردازش...",true);});
}
export default { render };