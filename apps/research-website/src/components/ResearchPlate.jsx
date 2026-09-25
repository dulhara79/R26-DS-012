export default function ResearchPlate(){
  return <svg className="research-plate" viewBox="0 0 760 500" role="img" aria-label="Scientific signal and evidence flow illustration">
    <rect x="1" y="1" width="758" height="498" fill="#fff" stroke="#d8d8d2"/>
    <text x="42" y="52" className="svg-label">MULTIMODAL EVIDENCE PLATE / 01</text>
    <path d="M44 152h55l15-38 20 76 20-63 18 25h84" className="signal"/>
    <text x="44" y="206" className="svg-small">PHYSIOLOGICAL / SHORT HORIZON</text>
    <g className="nodes"><circle cx="90" cy="302" r="7"/><circle cx="150" cy="270" r="7"/><circle cx="210" cy="320" r="7"/><circle cx="270" cy="282" r="7"/><path d="M90 302L150 270L210 320L270 282M150 270L270 282"/></g>
    <text x="44" y="365" className="svg-small">BEHAVIOURAL / LONGER HORIZON</text>
    <rect x="380" y="82" width="310" height="116" fill="#f7f7f4" stroke="#d8d8d2"/>
    <line x1="408" y1="116" x2="654" y2="116" className="thin"/><line x1="408" y1="142" x2="618" y2="142" className="thin"/><line x1="408" y1="168" x2="674" y2="168" className="thin"/>
    <text x="380" y="226" className="svg-small">CLINICAL NOTE / EMBEDDING EVIDENCE</text>
    <path d="M380 330h110M600 330h90" className="thin"/><circle cx="545" cy="330" r="44" fill="#2547b8"/><text x="545" y="325" textAnchor="middle" className="svg-white">FUSION</text><text x="545" y="345" textAnchor="middle" className="svg-white-small">RELIABILITY</text>
    <path d="M490 330h11M589 330h11M256 152C330 152 330 296 501 316M270 282C360 282 400 305 501 326" className="flow"/>
    <text x="380" y="420" className="svg-small">ELIGIBLE SIGNALS → WEIGHTING → EVIDENCE</text>
    <text x="380" y="452" className="svg-note">Unreliable or unavailable modalities can be withheld.</text>
  </svg>
}
