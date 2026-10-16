// Illustrative virtual trial: fixed 7.4 GBq x 4 cycles vs model-guided activity (kidney BED limit).
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x))}
const Phi=z=>0.5*(1+erf(z/Math.SQRT2));
function simulate(N=140, seed=7){
  const r=mulberry32(seed), randn=()=>{const u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
  const P={cycles:4, dK_med:0.5, dK_sd:0.35, dT_med:3.0, dT_sd:0.6, abK:2.5, GK:0.0231/(0.0231+0.2476), abT:10, GT:0.0099/(0.0099+0.462),
    L:28, Amax:12, Afix:7.4, alpha:0.15, TCP50:80, TD50:45, m:0.2};
  const N0=Math.LN2*Math.exp(P.alpha*P.TCP50);
  const bed=(D,G,ab)=>D*(1+G*D/ab);
  const out={fixed:[],guided:[],P};
  for(let i=0;i<N;i++){
    const dK=P.dK_med*Math.exp(P.dK_sd*randn()), dT=P.dT_med*Math.exp(P.dT_sd*randn());
    const a=P.GK/P.abK, x=(-1+Math.sqrt(1+4*a*P.L/P.cycles))/(2*a); // per-cycle kidney dose at limit
    const Aopt=Math.min(P.Amax, x/dK);
    for(const [arm,A] of [["fixed",P.Afix],["guided",Aopt]]){
      const bK=P.cycles*bed(A*dK,P.GK,P.abK), bT=P.cycles*bed(A*dT,P.GT,P.abT);
      const tcp=Math.exp(-N0*Math.exp(-P.alpha*bT)), ntcp=Phi((bK-P.TD50)/(P.m*P.TD50));
      out[arm].push({A,bK,bT,tcp,ntcp,pplus:tcp*(1-ntcp),over:bK>P.L});
    }
  }
  const mean=(a,k)=>a.reduce((s,o)=>s+(typeof o[k]==="boolean"?+o[k]:o[k]),0)/a.length;
  const sum={};
  for(const arm of ["fixed","guided"]){const a=out[arm];sum[arm]={A:mean(a,"A"),bK:mean(a,"bK"),bT:mean(a,"bT"),tcp:mean(a,"tcp"),ntcp:mean(a,"ntcp"),pplus:mean(a,"pplus"),over:mean(a,"over"),
    under:a.filter(o=>o.bK<0.6*P.L).length/a.length}}
  // Bayesian update of log kidney dose coefficient: prior sd 0.35, SPECT measurement sd 0.15
  const sp=0.35, sm=0.15, spost=Math.sqrt(1/(1/sp**2+1/sm**2));
  sum.bayes={prior:Math.exp(1.96*sp), post:Math.exp(1.96*spost), spost};
  return {sum,P};
}
module.exports={simulate};
if(require.main===module) console.log(JSON.stringify(simulate().sum,null,1));
