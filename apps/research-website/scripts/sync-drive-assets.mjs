// Sync the image slots and final PPTX from the shared Google Drive folder.
// Requires GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON and viewer access to the folders.
import fs from "node:fs/promises";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(root, "../src/data/driveAssets.generated.js");
const credentials = JSON.parse(process.env.GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON || "null");
if (!credentials?.client_email || !credentials?.private_key) {
  console.error("Set GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON to a service account JSON credential.");
  process.exit(1);
}

function base64url(v) { return Buffer.from(v).toString("base64url"); }
const now = Math.floor(Date.now() / 1000);
const head = base64url(JSON.stringify({alg:"RS256",typ:"JWT"}));
const payload = base64url(JSON.stringify({
  iss:credentials.client_email,scope:"https://www.googleapis.com/auth/drive.readonly",
  aud:credentials.token_uri || "https://oauth2.googleapis.com/token",iat:now,exp:now+3000
}));
const signature = crypto.sign("RSA-SHA256",Buffer.from(head+"."+payload),credentials.private_key).toString("base64url");
const auth = await fetch(credentials.token_uri || "https://oauth2.googleapis.com/token", {
  method:"POST", headers:{"content-type":"application/x-www-form-urlencoded"},
  body:new URLSearchParams({grant_type:"urn:ietf:params:oauth:grant-type:jwt-bearer",assertion:head+"."+payload+"."+signature})
});
if (!auth.ok) throw new Error("Drive authentication failed: "+auth.status);
const {access_token} = await auth.json();
async function list(folder) {
  let token, files=[];
  do {
    const params=new URLSearchParams({q:"'"+folder+"' in parents and trashed = false",fields:"nextPageToken,files(id,name,mimeType,modifiedTime,webContentLink)",pageSize:"1000"});
    if(token)params.set("pageToken",token);
    const res=await fetch("https://www.googleapis.com/drive/v3/files?"+params,{headers:{Authorization:"Bearer "+access_token}});
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      const reason = detail.error?.errors?.[0]?.reason || detail.error?.status || "unknown";
      const message = detail.error?.message || "No details returned by Google Drive API";
      const hint = res.status === 403
        ? "Check that the Google Drive API is enabled and the service account client_email has Viewer access to the shared folder and its contents."
        : "Verify the service-account configuration and Drive folder permissions.";
      throw new Error(`Drive listing failed (${folder}): HTTP ${res.status}; reason=${reason}; message=${message}. ${hint}`);
    }
    const data=await res.json();files.push(...(data.files||[]));token=data.nextPageToken;
  }while(token);
  return files;
}
const imageRoot="1xzor8mCUxMeKh1-f12dOhAcWCNccQzgJ";
const dirMap={
  components:{c1:"c1-wearable.webp",c2:"c2-behaviour.webp",c3:"c3-clinical-notes.webp",c4:"c4-fusion.webp"},
  banners:{bannerResearch:"research.webp",bannerComponents:"components.webp",bannerSystem:"system.webp",bannerFindings:"findings.webp",bannerDocuments:"documents.webp",bannerTeam:"team.webp",bannerContact:"contact.webp"},
  system:{appAura:"aura-app.webp",appClinAnx:"clinanx-console.webp"},
  team:{"team-c1":"sendanayake.jpg","team-c2":"layathma.jpg","team-c3":"kaushalya.jpg","team-c4":"seneviratne.jpg","sup-1":"thelijjagoda.jpg","sup-2":"weerasinghe.jpg","sup-3":"suraweera.jpg"},
  home:{homeContext:"context.webp"}
};
const directories=await list(imageRoot);
const manifest={};
for(const [directory,slots] of Object.entries(dirMap)){
  const child=directories.find(f=>f.mimeType==="application/vnd.google-apps.folder"&&f.name===directory);
  if(!child)throw new Error("Missing images folder: "+directory);
  const files=await list(child.id);
  for(const [slot,filename] of Object.entries(slots)){
    const candidates=files.filter(f=>f.name===filename&&f.mimeType.startsWith("image/"));
    if(!candidates.length)throw new Error("Missing Drive image: "+directory+"/"+filename);
    candidates.sort((a,b)=>(b.modifiedTime||"").localeCompare(a.modifiedTime||"")||a.id.localeCompare(b.id));
    manifest[slot]={id:candidates[0].id,modifiedTime:candidates[0].modifiedTime};
  }
}
const finalFolder="1Bah9NZqTjrcwa1GyZV8lPIIqN0w-AxkX";
const presentations=(await list(finalFolder)).filter(f=>/\.pptx$/i.test(f.name)&&f.mimeType==="application/vnd.openxmlformats-officedocument.presentationml.presentation");
if(!presentations.length)throw new Error("No PPTX present in the verified final presentation folder");
presentations.sort((a,b)=>(b.modifiedTime||"").localeCompare(a.modifiedTime||"")||a.id.localeCompare(b.id));
const final={id:presentations[0].id,name:presentations[0].name,modifiedTime:presentations[0].modifiedTime};
const next="// Generated from the project's linked Google Drive assets; do not edit by hand.\n"+
  "// Regenerate with: npm run drive:sync (requires a Drive read-only service account).\n"+
  "export const driveImageFiles = Object.freeze("+JSON.stringify(manifest,null,2)+");\n\n"+
  "export const driveFinalPresentation = Object.freeze("+JSON.stringify(final,null,2)+");\n";
const old=await fs.readFile(output,"utf8").catch(()=>"");
if(old!==next){await fs.writeFile(output,next);console.log("Drive assets updated");}
else console.log("Drive assets unchanged");
