"use strict";
const $ = id => document.getElementById(id);
const data = window.SCALESCOUT_DATA || {summary:null,cases:[],names:[]};
const labels = {global:"全图检测",uniform:"均匀覆盖",random:"固定种子随机",adaptive:"线索与探索",evidence:"仅检测线索",texture:"仅纹理细节",full:"完整切片"};
let current = null, photo = null, photoURL = null, imported = false;
function option(parent,value,text){const el=document.createElement("option");el.value=value;el.textContent=text;parent.append(el);}
function cell(row,value){const el=document.createElement("td");el.textContent=value;row.append(el);}
function valid(trace){return trace && Number.isFinite(trace.width) && trace.width>0 && trace.width<=16384 && Number.isFinite(trace.height) && trace.height>0 && trace.height<=16384 && Array.isArray(trace.tiles) && trace.tiles.every(t=>Array.isArray(t.box)&&t.box.length===4&&t.box.every(Number.isFinite)) && Array.isArray(trace.detections) && trace.detections.every(d=>Array.isArray(d.box)&&d.box.length===4&&d.box.every(Number.isFinite)&&Number.isFinite(d.score));}
function draw(){
  if(!current)return;
  const canvas=$("canvas"),ctx=canvas.getContext("2d");
  canvas.width=current.width;canvas.height=current.height;
  ctx.fillStyle="#0c1720";ctx.fillRect(0,0,canvas.width,canvas.height);
  if(photo)ctx.drawImage(photo,0,0,canvas.width,canvas.height);
  else{ctx.strokeStyle="#1f303e";ctx.lineWidth=1;for(let x=0;x<canvas.width;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}for(let y=0;y<canvas.height;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}}
  const line=Math.max(1.5,canvas.width/300);
  for(const t of current.tiles){
    if(!Array.isArray(t.box)||!t.box.every(Number.isFinite))continue;
    if((t.selected&&!$("tiles").checked)||(!t.selected&&!$("skipped").checked))continue;
    const [x,y,x2,y2]=t.box;
    ctx.strokeStyle=t.selected?"#ffc77b":"#526574";ctx.lineWidth=line;ctx.setLineDash(t.selected?[]:[5,5]);ctx.strokeRect(x,y,x2-x,y2-y);ctx.setLineDash([]);
    if(t.selected){ctx.fillStyle="#ffc77b";ctx.font=`${Math.max(12,canvas.width/40)}px sans-serif`;ctx.fillText(`区域 ${t.index} · 第${t.order+1}次`,x+5,y+20);}
  }
  if($("boxes").checked)for(const d of current.detections){
    const [x,y,x2,y2]=d.box;ctx.strokeStyle="#69e3c2";ctx.lineWidth=line;ctx.strokeRect(x,y,x2-x,y2-y);
    ctx.font=`${Math.max(11,canvas.width/48)}px sans-serif`;ctx.fillStyle="#69e3c2";
    ctx.fillText(`${data.names[d.class_id]||d.class_id} ${d.score.toFixed(2)}`,x+3,Math.max(12,y-4));
  }
}
function render(trace){
  if(!valid(trace))throw new Error("trace 格式无效：需要原图尺寸、区域和检测框");
  current=trace;
  $("calls").textContent=trace.detector_calls;$("objects").textContent=trace.detections.length;
  $("sample-title").textContent=trace.image_name||"导入的预测";$("size").textContent=`${trace.width} × ${trace.height} px`;
  $("detail").textContent=`策略：${labels[trace.config?.strategy]||trace.config?.strategy||"导入"}。候选区域 ${trace.tiles.length} 个，选中 ${trace.tiles.filter(t=>t.selected).length} 个。全图低阈值候选 ${trace.global_proposals?.length||0} 个。`;
  $("tile-table").replaceChildren();
  for(const t of [...trace.tiles].sort((a,b)=>(a.selected?0:1)-(b.selected?0:1)||(a.order??999)-(b.order??999)||a.index-b.index)){
    const row=document.createElement("tr");if(t.selected)row.className="selected";
    [t.index,t.box.join(", "),Number(t.texture).toFixed(4),Number(t.evidence).toFixed(4),t.selected?`第 ${t.order+1} 次复查`:"未使用预算"].forEach(v=>cell(row,v));$("tile-table").append(row);
  }
  draw();
}
function changeCase(){
  imported=false;photo=null;$("image-status").textContent="原图没有随仓库分发。文件只在本地读取。";
  const entry=data.cases[Number($("case").value)];if(!entry)return;
  const before=$("strategy").value;$("strategy").replaceChildren();
  for(const strategy of Object.keys(entry.traces))option($("strategy"),strategy,labels[strategy]||strategy);
  $("strategy").value=entry.traces[before]?before:"adaptive";render(entry.traces[$("strategy").value]);
}
data.cases.forEach((entry,i)=>option($("case"),String(i),entry.label||entry.image));
$("case").addEventListener("change",changeCase);
$("strategy").addEventListener("change",()=>{if(!imported)render(data.cases[Number($("case").value)].traces[$("strategy").value]);});
["tiles","boxes","skipped"].forEach(id=>$(id).addEventListener("change",draw));
$("trace-file").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file)return;
  try{const trace=JSON.parse(await file.text());photo=null;render(trace);imported=true;$("image-status").textContent="预测已导入，可载入对应原图。";}catch(error){$("image-status").textContent=error.message;}
});
$("image-file").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file||!current)return;
  if(photoURL)URL.revokeObjectURL(photoURL);photoURL=URL.createObjectURL(file);
  const image=new Image();image.onload=async()=>{
    if(image.naturalWidth!==current.width||image.naturalHeight!==current.height){$("image-status").textContent="原图尺寸不匹配，请载入该预测对应的图片。";photo=null;draw();return;}
    if(current.input_sha256 && window.crypto?.subtle){
      const bytes=await crypto.subtle.digest("SHA-256",await file.arrayBuffer());
      const digest=Array.from(new Uint8Array(bytes)).map(b=>b.toString(16).padStart(2,"0")).join("");
      if(digest!==current.input_sha256){$("image-status").textContent="文件哈希不匹配，请选择生成该预测的原图。";photo=null;draw();return;}
    }
    photo=image;$("image-status").textContent="原图已在本地载入。";draw();
  };image.onerror=()=>{$("image-status").textContent="无法读取图片。";};image.src=photoURL;
});
const pct=x=>x==null?"—":`${(100*x).toFixed(2)}%`;
if(data.summary){
  $("protocol").textContent=`${data.summary.sample_size} 张验证图片 · ${data.summary.environment.hardware} · K=${data.summary.budget_extra_crops}`;
  for(const [strategy,m] of Object.entries(data.summary.summary)){
    const row=document.createElement("tr");if(strategy==="adaptive")row.className="featured";
    [labels[strategy]||strategy,m.mean_calls.toFixed(1),pct(m.precision50),pct(m.recall50),pct(m.small_recall50),`${m.median_ms.toFixed(1)} ms`].forEach(v=>cell(row,v));$("metrics").append(row);
  }
}
if(data.cases.length)changeCase();else $("image-status").textContent="尚无内置实验数据，请导入 trace.json。";
