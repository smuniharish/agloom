import React, {FormEvent, useEffect, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";
import keycloak, {authenticatedFetch, initializeAuth, streamSse} from "./auth";

type Approval = {
  action_id:string; tool:string; policy_rule:string; expires_at?:string;
  decision?:"approve"|"reject"; decided_by?:string; decided_at?:string; note?:string;
};
type Run = {
  id:string; objective:string; mode:"loop"|"one_shot";
  status:"running"|"awaiting_approval"|"approved"|"rejected"|"unvalidated"|"failed"|"cancelled";
  execution_status:"queued"|"running"|"paused"|"completed"|"failed"|"cancelled";
  objective_state:string; termination?:string; error?:string; published_at?:string;
  accuracy:{quality:number;coverage:number;citation_precision:number;gaps:string[];violations:string[]};
  timeline:{iteration:number;action:string;progress:number;quality:number;gaps:string[];violations:string[];reason:string}[];
  synthesis?:{summary:string;claims:{id:string;topic:string;text:string;status:string;citations:{source_id:string;quote:string}[]}[]};
  sources:{id:string;title:string;revision:string;uri:string}[];
  approval?:Approval;
};
type RunEvent = {
  id?:number;
  type:"queued"|"started"|"resumed"|"runtime"|"iteration"|"awaiting_approval"|"decision"|"published"|"retrying"|"cancelling"|"cancelled"|"completed"|"failed";
  timestamp:string; message?:string; name?:string; iteration?:number;
  progress?:number; quality?:number; action?:string; run?:Run;
};

const STREAM_END=new Set(["awaiting_approval","completed","failed","cancelled"]);
const ACTIVE=new Set(["queued","running"]);
function pct(value:number){return `${Math.round(value*100)}%`}
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));

function App(){
  const savedRunKey=`agloom:evidence-lab:${keycloak.tokenParsed?.tenant_id}:${keycloak.subject}:run`;
  const [objective,setObjective]=useState("Verify Acme's current security, incident response, and compliance posture.");
  const [topics,setTopics]=useState("encryption,incident response,SOC 2");
  const [target,setTarget]=useState("http://target-site");
  const [mode,setMode]=useState<"loop"|"one_shot">("loop");
  const [run,setRun]=useState<Run|null>(null);
  const [events,setEvents]=useState<RunEvent[]>([]);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const lastEventId=useRef("");
  useEffect(()=>{
    const runId=sessionStorage.getItem(savedRunKey);
    if(!runId)return;
    const controller=new AbortController();
    let mounted=true;
    setBusy(true);
    authenticatedFetch(`/api/runs/${runId}`).then(async response=>{
      if(response.status===403||response.status===404){
        sessionStorage.removeItem(savedRunKey);
        throw new Error("Saved run is no longer available for this account.");
      }
      if(!response.ok)throw new Error(await response.text());
      const current:Run=await response.json();
      if(mounted){setRun(current);await streamRun(runId,controller)}
    }).catch(cause=>{if(mounted)setError(String(cause))})
      .finally(()=>{if(mounted)setBusy(false)});
    return ()=>{mounted=false;controller.abort()};
  },[savedRunKey]);
  async function pollRun(runId:string){
    for(;;){
      const response=await authenticatedFetch(`/api/runs/${runId}`);
      if(!response.ok) throw new Error(await response.text());
      const current:Run=await response.json();
      setRun(current);
      if(!ACTIVE.has(current.execution_status)) return current;
      await delay(1500);
    }
  }
  function streamRun(runId:string,controller=new AbortController()){
    return new Promise<Run>((resolve,reject)=>{
      streamSse(`/api/runs/${runId}/events`,(event:RunEvent)=>{
        if(event.id!==undefined)lastEventId.current=String(event.id);
        setEvents(current=>[...current.slice(-49),event]);
        if(event.run)setRun(event.run);
        if(STREAM_END.has(event.type)){
          controller.abort();
          if(event.run)resolve(event.run);else pollRun(runId).then(resolve,reject);
        }
      },controller.signal,lastEventId.current).then(()=>{
        if(controller.signal.aborted)reject(new DOMException("Run stream closed","AbortError"));
      }).catch(cause=>{
        if(controller.signal.aborted)reject(cause);
        else pollRun(runId).then(resolve,reject);
      });
    });
  }
  async function submit(event:FormEvent){
    event.preventDefault(); setBusy(true); setError(""); setEvents([]); setRun(null);
    lastEventId.current="";
    try {
      const started=await authenticatedFetch("/api/runs",{method:"POST",headers:{"Content-Type":"application/json","Idempotency-Key":crypto.randomUUID()},
        body:JSON.stringify({objective,topics:topics.split(",").map(x=>x.trim()).filter(Boolean),target_url:target||null,mode})});
      if(!started.ok) throw new Error(await started.text());
      const {run_id}=await started.json();
      sessionStorage.setItem(savedRunKey,run_id);
      await streamRun(run_id);
    } catch(e){setError(String(e))} finally {setBusy(false)}
  }
  async function decide(decision:"approve"|"reject"){
    if(!run)return;
    setBusy(true); setError("");
    try {
      const response=await authenticatedFetch(`/api/runs/${run.id}/decision`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision})});
      if(!response.ok) throw new Error(await response.text());
      const body:{run:Run}=await response.json();
      setRun(body.run);
      await streamRun(run.id);
    } catch(e){setError(String(e))} finally {setBusy(false)}
  }
  async function cancel(){
    if(!run)return;
    const response=await authenticatedFetch(`/api/runs/${run.id}/cancel`,{method:"POST"});
    if(!response.ok){setError(await response.text());return}
    const {status}=await response.json();
    if(status==="cancelled")await pollRun(run.id);
  }
  const reviewPending=run?.status==="awaiting_approval"&&!run.approval?.decision;
  const cancellable=run&&["queued","running","paused"].includes(run.execution_status)&&!["approved","rejected"].includes(run.status);
  return <main>
    <header><div className="eyebrow">AGLOOM · LOOP ENGINEERING</div><h1>Continuous Evidence Lab</h1>
      <p>Refresh, retrieve, cite, validate, replan — then hold the evidence for an accountable reviewer before it is published.</p>
      <small>Signed in as {keycloak.subject} · tenant {String(keycloak.tokenParsed?.tenant_id)} · <button onClick={()=>keycloak.logout()}>Logout</button></small></header>
    <section className="panel input"><form onSubmit={submit}>
      <label>Research objective<textarea value={objective} onChange={e=>setObjective(e.target.value)} required minLength={8}/></label>
      <div className="formgrid"><label>Required topics<input value={topics} onChange={e=>setTopics(e.target.value)}/></label>
      <label>Live target (optional)<input type="url" value={target} onChange={e=>setTarget(e.target.value)}/></label></div>
      <label>Execution mode<select value={mode} onChange={e=>setMode(e.target.value as "loop"|"one_shot")}>
        <option value="loop">Loop Engineering</option><option value="one_shot">One-shot baseline</option>
      </select></label>
      <button disabled={busy}>{busy?"Running bounded evidence loop…":"Run synthesis"}</button>
      {cancellable&&<button type="button" className="secondary" onClick={cancel}>Cancel run</button>}
      {error&&<p role="alert" className="error">{error}</p>}
    </form></section>
    {events.length>0&&<section className="panel live" aria-live="polite">
      <div className="live-head"><div><span className="eyebrow">LIVE RUN STREAM</span><h2>Status &amp; progress</h2></div>
        <span className={busy?"connection active":"connection"}>{busy?"CONNECTED":run?.execution_status.toUpperCase()}</span></div>
      <ol>{events.map((event,index)=><li key={`${event.id??event.timestamp}-${index}`}>
        <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
        <b>{event.type==="runtime"?event.name:event.type}</b>
        <span>{event.message??event.action??(event.iteration?`Iteration ${event.iteration}`:"Runtime event")}</span>
        {event.progress!==undefined&&<em>{pct(event.progress)}</em>}
      </li>)}</ol>
    </section>}
    {run&&<><section className="metrics">
      {[["Quality",run.accuracy.quality],["Coverage",run.accuracy.coverage],["Citation precision",run.accuracy.citation_precision]].map(([name,value])=>
        <article className="metric" key={String(name)}><span>{name}</span><strong>{pct(Number(value))}</strong><div><i style={{width:pct(Number(value))}}/></div></article>)}
      <article className="metric state"><span>Objective state</span><strong>{run.objective_state.replace("_"," ")}</strong><small>{run.status.replace("_"," ")} · {run.termination??run.execution_status}</small></article>
    </section>
    {run.error&&<section className="panel error"><h2>Run {run.status==="cancelled"?"cancelled":"failed"}</h2><p>{run.error}</p></section>}
    <section className="layout"><article className="panel"><h2>Iteration timeline</h2>
      <ol className="timeline">{run.timeline.map(item=><li key={item.iteration}><b>{item.iteration}</b><div><span className="tag">{item.action}</span>
        <strong>{pct(item.quality)} quality · {pct(item.progress)} progress</strong><p>{item.reason}</p>
        {item.gaps.length>0&&<small>Gaps: {item.gaps.join(", ")}</small>}</div></li>)}</ol></article>
      <article className="panel"><h2>{run.published_at?"Published synthesis":"Validated synthesis"}</h2><p>{run.synthesis?.summary}</p>
        {run.synthesis?.claims.map(claim=><details key={claim.id}><summary><span className={`dot ${claim.status}`}/>{claim.topic}: {claim.text}</summary>
          {claim.citations.map((cite,i)=><blockquote key={i}><code>{cite.source_id}</code> “{cite.quote}”</blockquote>)}</details>)}
        <h3>Open gaps & violations</h3>{[...run.accuracy.gaps,...run.accuracy.violations].length?
          <ul>{[...run.accuracy.gaps,...run.accuracy.violations].map(x=><li key={x}>{x}</li>)}</ul>:<p className="good">Deterministic validation found no open gaps.</p>}
      </article></section>
      {run.approval&&<section className="panel approval"><div>
        <h2>{reviewPending?"Human approval gate":"Review decision"}</h2>
        {reviewPending?<p>Publication is held by the policy rule “{run.approval.policy_rule}”{run.approval.expires_at?` until ${new Date(run.approval.expires_at).toLocaleString()}`:""}. Approving publishes exactly this synthesis once.</p>
          :<p>{run.approval.decision==="approve"?"Approved":"Rejected"} by {run.approval.decided_by}{run.published_at?` · published ${new Date(run.published_at).toLocaleString()}`:run.approval.decision==="approve"?" · publishing…":""}.</p>}</div>
        {reviewPending&&<div><button className="secondary" disabled={busy} onClick={()=>decide("reject")}>Reject</button>
        <button disabled={busy} onClick={()=>decide("approve")}>Approve and publish</button></div>}</section>}
      <section className="panel"><h2>Evidence ledger</h2><div className="sources">{run.sources.map(s=><article key={s.id}><code>{s.id}</code><b>{s.title}</b><small>rev {s.revision} · {s.uri}</small></article>)}</div></section>
    </>}
  </main>
}
initializeAuth()
  .then(()=>createRoot(document.getElementById("root")!).render(<App/>))
  .catch(error=>{document.getElementById("root")!.textContent=`Authentication failed: ${String(error)}`});
