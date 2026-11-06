'use strict';
const $ = id => document.getElementById(id);
let schema, data, dirty = false, playing = false, seconds = 0, last = 0, running = false;
const names = ['q1', 'q2', 'dq1', 'dq2'];
const pretty = (value, digits=3) => value == null ? 'Unavailable' : Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});

function notice(message, kind='') { $('notice').textContent=message; $('notice').className=kind; }
function values() {
  const settings={};
  schema.groups.forEach(group=>group.fields.forEach(field=>{
    const raw=$('p-'+field.key).value;
    settings[field.key]=field.type==='choice' ? raw : (raw.trim()==='' ? null : Number(raw));
  }));
  return settings;
}
function apply(settings) {
  for (const [key,value] of Object.entries(settings)) { const el=$('p-'+key); if(el) el.value=value; }
  derived();
}
function derived() {
  const target=$('derived');
  if(!target) return;
  const s=values();
  target.textContent=`Derived automatically: COM₁ = ${pretty(s.l1/2)} m · COM₂ = ${pretty(s.l2/2)} m. Inertia I₁ = ${pretty(s.m1*s.l1*s.l1/12,4)} kg·m² · I₂ = ${pretty(s.m2*s.l2*s.l2/12,4)} kg·m².`;
  const sine=s.initial_guess==='sine';
  $('p-amplitude').disabled=!sine; $('p-period').disabled=!sine;
}
function markDirty() {
  dirty=true;
  $('dirty').textContent='Changes not applied · click Run simulation';
  if(!running) notice('Parameters changed. The animation still shows the previous run. Click Run simulation to apply your settings.');
  derived();
}
function buildForm() {
  for (const [index,group] of schema.groups.entries()) {
    const details=document.createElement('details'); details.open=index===0;
    const summary=document.createElement('summary'); summary.textContent=group.title;
    const note=document.createElement('p'); note.className='group-note'; note.textContent=group.note;
    details.append(summary,note);
    group.fields.forEach(field=>{
      const wrapper=document.createElement('div'); wrapper.className='field';
      const label=document.createElement('label'); label.htmlFor='p-'+field.key; label.textContent=field.label;
      if(field.unit) {const unit=document.createElement('span');unit.textContent=field.unit;label.append(unit);}
      let input;
      if(field.type==='choice') {
        input=document.createElement('select');
        field.options.forEach(([value,text])=>{const option=document.createElement('option');option.value=value;option.textContent=text;input.append(option);});
      } else {
        input=document.createElement('input'); input.type='number'; input.min=field.min; input.max=field.max;
        // Any real value within the bounds is valid; integer fields use step 1.
        input.step=field.type==='integer'?'1':'any'; input.required=true;
      }
      input.id='p-'+field.key; input.name=field.key; input.value=field.default;
      input.addEventListener('input',markDirty);
      wrapper.append(label,input);
      if(field.help) {const help=document.createElement('small');help.textContent=field.help;wrapper.append(help);}
      details.append(wrapper);
    });
    if(group.title==='Robot') {const d=document.createElement('div');d.id='derived';d.className='derived';details.append(d);}
    $('groups').append(details);
  }
}
function setBusy(busy) {
  running=busy; $('run').disabled=busy; $('reset').disabled=busy;
  $('run').textContent=busy?'Computing trajectory…':'▶  Run simulation';
  $('parameters').querySelectorAll('input,select').forEach(el=>el.disabled=busy);
  if(!busy) derived();
}
function show(result) {
  if(!result) return;
  data=result; seconds=0; playing=false; $('play').textContent='▶ Play';
  const row=data.comparison.find(r=>r.method===data.selected);
  $('objective').textContent=pretty(row.cost,5);
  $('optimizer').textContent=`${row.method==='ddp'?'Full DDP':'iLQR'} · ${row.iterations} iterations`;
  $('torque').textContent=pretty(data.execution.peak_torque,2)+' N·m';
  for(const key of ['swing','hold']) {
    const pass=data.execution[key==='swing'?'swing_success':'hold_success'];
    $(key).textContent=pass?'Passed':'Failed';$(key).className=pass?'pass':'fail';
  }
  $('example').textContent=data.example?'Verified example':'Computed run';
  $('comparison').replaceChildren();
  data.comparison.forEach(r=>{
    const tr=document.createElement('tr');
    [r.method==='ddp'?'Full DDP':'iLQR',pretty(r.cost,6),r.iterations,pretty(r.seconds,2)+' s',r.status.replaceAll('_',' '),r.terminal_error.toExponential(2)].forEach(text=>{
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    });$('comparison').append(tr);
  });
  const s=data.settings;
  $('run-description').textContent=`Shown run: ${s.duration} s swing-up + ${s.hold} s local LQR. `+
    `${s.policy==='feedback'?'Trajectory feedback':'Open loop'}; ${s.integration==='fine'?'10 integration substeps per sample':'same RK4 model as the optimizer'}. `+
    `Physical success uses angle errors modulo 2π. ${data.execution.terminated?'Execution was stopped after numerical divergence. ':''}`+
    (data.comparison.length>1?`Both optimizers used identical settings and the same initial guess; the lower-cost ${data.selected.toUpperCase()} solution is animated.`:'');
  $('timeline').max=data.execution.x.length-1;$('timeline').value=0;
  draw();
}
function drawRobot(index) {
  const canvas=$('robot'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  c.clearRect(0,0,w,h);c.fillStyle='#112632';c.fillRect(0,0,w,h);
  c.strokeStyle='#ffffff07';c.lineWidth=1;
  for(let x=20;x<w;x+=40){c.beginPath();c.moveTo(x,60);c.lineTo(x,h-50);c.stroke();}
  for(let y=70;y<h-40;y+=40){c.beginPath();c.moveTo(50,y);c.lineTo(w-50,y);c.stroke();}
  if(!data)return;
  const s=data.settings,x=data.execution.x[index],origin=[w/2,h/2+10],scale=180/(s.l1+s.l2);
  c.strokeStyle='#ffffff13';c.beginPath();c.moveTo(origin[0],70);c.lineTo(origin[0],h-55);c.stroke();
  const tip=(p,q,l)=>[p[0]+scale*l*Math.sin(q),p[1]+scale*l*Math.cos(q)];
  const ghost1=tip(origin,Math.PI,s.l1),ghost2=tip(ghost1,Math.PI,s.l2);
  c.save();c.setLineDash([6,7]);c.strokeStyle='#92b6c655';c.lineWidth=3;
  c.beginPath();c.moveTo(...origin);c.lineTo(...ghost1);c.lineTo(...ghost2);c.stroke();c.restore();
  const p1=tip(origin,x[0],s.l1),p2=tip(p1,x[0]+x[1],s.l2);
  c.lineCap='round';c.lineWidth=13;
  [[origin,p1,'#4ec7ef'],[p1,p2,'#fcaa67']].forEach(([a,b,color])=>{c.strokeStyle=color;c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();});
  [origin,p1,p2].forEach((p,i)=>{c.fillStyle=i===0?'#e9f6fb':'#fff';c.beginPath();c.arc(...p,i===0?10:7,0,Math.PI*2);c.fill();});
  c.strokeStyle='#ffffff2a';c.lineWidth=2;c.beginPath();c.arc(...origin,19,0,Math.PI*2);c.stroke();
  c.fillStyle='#668a9d';c.font='12px Segoe UI';c.textAlign='center';c.fillText('fixed motor base',origin[0],origin[1]+38);
  $('clock').textContent=(index*s.dt).toFixed(2)+' s';
  $('phase').textContent=index*s.dt>=s.duration?'Upright hold · local LQR':(s.policy==='feedback'?'Swing-up · trajectory feedback':'Swing-up · open loop');
  $('state-values').textContent=`q1 = ${x[0].toFixed(3)} · q2 = ${x[1].toFixed(3)} rad`;
}
function drawCharts(index) {
  const canvas=$('charts'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  c.clearRect(0,0,w,h);if(!data)return;
  const e=data.execution,s=data.settings,left=56,right=w-18,total=s.duration+s.hold;
  const tx=t=>left+(right-left)*t/total;
  const bands=[{top:23,bottom:121,label:'Angles (rad)',series:[e.x.map(x=>x[0]),e.x.map(x=>x[1])],colors:['#0a91b5','#df8b45']},
    {top:168,bottom:239,label:'Torque (N·m)',series:[e.u],colors:['#0f817a']}];
  bands.forEach((b,bi)=>{
    let lo=0,hi=bi===0?Math.PI:1;
    b.series.forEach(arr=>arr.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
    const pad=(hi-lo)*.13;lo-=pad;hi+=pad;
    const ty=v=>b.bottom-(v-lo)/(hi-lo)*(b.bottom-b.top);
    c.font='11px Segoe UI';c.fillStyle='#768b99';c.textAlign='left';c.fillText(b.label,left,bi===0?12:156);
    c.textAlign='right';c.fillText(pretty(hi,1),left-8,b.top+6);c.fillText(pretty(lo,1),left-8,b.bottom);
    c.strokeStyle='#e6edf1';c.lineWidth=1;
    for(const y of [b.top,b.bottom]){c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();}
    c.save();c.beginPath();c.rect(left,b.top,right-left,b.bottom-b.top);c.clip();
    function line(arr,color,opacity=1) {c.globalAlpha=opacity;c.strokeStyle=color;c.lineWidth=1.8;c.beginPath();arr.forEach((v,k)=>k?c.lineTo(tx(k*s.dt),ty(v)):c.moveTo(tx(0),ty(v)));c.stroke();c.globalAlpha=1;}
    if(bi===0){line(data.nominal_x.map(x=>x[0]),b.colors[0],.2);line(data.nominal_x.map(x=>x[1]),b.colors[1],.2);}
    b.series.forEach((arr,i)=>line(arr,b.colors[i]));
    c.setLineDash([4,4]);c.strokeStyle='#9aadb8';c.beginPath();c.moveTo(tx(s.duration),b.top);c.lineTo(tx(s.duration),b.bottom);c.stroke();
    c.setLineDash([]);c.strokeStyle='#304f6566';c.beginPath();c.moveTo(tx(index*s.dt),b.top);c.lineTo(tx(index*s.dt),b.bottom);c.stroke();c.restore();
  });
  c.textAlign='center';c.fillStyle='#8598a3';c.font='10px Segoe UI';
  for(let t=0;t<=total;t+=Math.max(1,Math.ceil(total/7)))c.fillText(t+' s',tx(t),h-3);
}
function draw(){if(!data)return;const i=Math.min(data.execution.x.length-1,Math.round(seconds/data.settings.dt));$('timeline').value=i;drawRobot(i);drawCharts(i);}
function animate(now){
  if(playing&&data){seconds+=(now-last)/1000*Number($('speed').value);const end=(data.execution.x.length-1)*data.settings.dt;if(seconds>=end){seconds=end;playing=false;$('play').textContent='▶ Play';}draw();}
  last=now;requestAnimationFrame(animate);
}
async function poll(){
  try{
    const res=await fetch('/api/status');if(!res.ok)throw Error('Connection unavailable.');const status=await res.json();
    if(status.phase==='running'){setBusy(true);notice(status.message+' You can keep viewing the previous animation.','busy');setTimeout(poll,800);}
    else {setBusy(false);if(status.phase==='done'){show(await (await fetch('/api/result')).json());dirty=false;$('dirty').textContent='Settings applied · result saved';notice('Simulation finished. Press Play to view the new result.'+(data.execution.terminated?' Execution failed before the requested end time.':''),data.execution.terminated?'error':'');}
      else if(status.phase==='error')notice(status.message+' Adjust the parameters and try again.','error');}
  }catch(error){setBusy(false);notice('Connection lost. Reopen Launch Pendubot.cmd to reconnect.','error');}
}
$('parameters').addEventListener('submit',async event=>{
  event.preventDefault();if(running||!$('parameters').reportValidity())return;
  setBusy(true);notice('Starting the simulation…','busy');
  try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-App-Token':document.querySelector('meta[name="app-token"]').content},body:JSON.stringify(values())});
    const payload=await response.json();if(!response.ok)throw Error(payload.error);poll();
  }catch(error){setBusy(false);notice(error.message,'error');}
});
$('reset').onclick=()=>{apply(schema.defaults);markDirty();};
$('play').onclick=()=>{if(!data)return;const end=(data.execution.x.length-1)*data.settings.dt;if(seconds>=end)seconds=0;playing=!playing;$('play').textContent=playing?'Ⅱ Pause':'▶ Play';};
$('restart').onclick=()=>{seconds=0;draw();};
$('timeline').oninput=()=>{if(!data)return;seconds=Number($('timeline').value)*data.settings.dt;draw();};
$('export').onclick=()=>{if(!data)return;const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='pendubot-run.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
(async()=>{try{schema=await (await fetch('/api/schema')).json();buildForm();const result=await (await fetch('/api/result')).json();show(result);apply(result?.settings||schema.defaults);notice(result?.example?'Verified default run loaded. Press Play, or change the parameters and run a new experiment.':'Your last completed run is loaded. Change the parameters to explore another experiment.');poll();requestAnimationFrame(animate);}catch(error){notice('Could not load the simulation. Reopen Launch Pendubot.cmd.','error');}})();
