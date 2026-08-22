/**
 * ================================================================
 * ANXIETY RESEARCH DATA LOGGER — Privacy-Hardened Version
 * SLIIT Research Study | Ethical Approval Compliant
 * ================================================================
 */

var ROTATED_TABS = ["Raw", "EMA", "Physio"];
var DATATYPE_TO_FAMILY = {
  "EMA_Rating_morning":   "EMA",
  "EMA_Rating_afternoon": "EMA",
  "EMA_Rating_evening":   "EMA",
  "GAD7_Weekly":          "GAD7",
  "PSS10_Weekly":         "PSS10",
  "Demographics":         "Demographics",
  "Consent_Record":       "Consent_Log",
  "Consent_Withdrawal":   "Consent_Log",
  "Data_Deletion_Request": "Consent_Log",
  "Data_Export_Request":  "Consent_Log",
  "Physio_Vitals":        "Physio",
};

var HEADERS = {
  "Raw": ["Timestamp","Date","Time","Participant ID","Data Type","Value"],
  "EMA": ["Timestamp","Date","Time","Participant ID","Period",
          "Stress","Anxiety","Fatigue","Social","Activity Context","Raw JSON"],
  "Physio": ["Timestamp","Date","Time","Participant ID",
             "Heart Rate","Breathing Rate","Body Temp",
             "Motion Magnitude","Risk Score","Risk Label","Raw JSON"],
  "GAD7": ["Timestamp","Date","Participant ID","Total Score (0-21)","Severity",
           "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Raw JSON"],
  "PSS10": ["Timestamp","Date","Participant ID","Total Score (0-40)",
            "Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","Raw JSON"],
  "Demographics": ["Timestamp","Date","Participant ID","Age","Gender",
                   "Marital Status","Employment","Financial Status","Education",
                   "Living Situation","Anxiety Diagnosis","On Medication",
                   "Sleep Quality","Raw JSON"],
  "Consent_Log": ["Timestamp","Date","Participant ID","Event Type",
                  "Consent Version","Details","Raw JSON"],
  "Errors": ["Timestamp","Participant ID","Data Type","Raw Value","Error"],
  "Summary": []
};

var HEADER_BG    = "#00695C";
var HEADER_FG    = "#FFFFFF";
var GPS_PRECISION = 0.01;
var MAX_ROWS_PER_POST = 500;

// ════════════════════════════════════════════════════════════
// ENTRY POINTS
// ════════════════════════════════════════════════════════════

function doGet(e) {
  var props    = PropertiesService.getScriptProperties();
  var token    = props.getProperty("AUTH_TOKEN");
  var received = e && e.parameter && e.parameter.token;
  if (token && received !== token) return jsonErr("Unauthorized");
  return jsonOk({ status: "healthy", study: "SLIIT Anxiety Research" });
}

function doPost(e) {
  var props = PropertiesService.getScriptProperties();

  // 1. Parse
  var entries;
  try {
    var raw = (e && e.postData && e.postData.contents)
      ? e.postData.contents : JSON.stringify(e.parameter || {});
    var payload = JSON.parse(raw);
    entries = Array.isArray(payload) ? payload : [payload];
  } catch (err) {
    return jsonErr("Parse error: " + err);
  }

  if (!entries || entries.length === 0)
    return jsonOk({ status: "success", written: 0 });

  // 2. Rate limit
  if (entries.length > MAX_ROWS_PER_POST)
    return jsonErr("Batch too large. Max " + MAX_ROWS_PER_POST + " per request.");

  // 3. Auth
  var authToken = props.getProperty("AUTH_TOKEN");
  if (authToken) {
    var received = entries[0] && entries[0].token;
    if (received !== authToken) {
      console.warn("[AUTH FAIL] hash=" + hashString(String(received || "")));
      return jsonErr("Unauthorized");
    }
    entries = entries.map(stripToken);
  }

  // 4. Sanitize
  entries = entries.map(sanitizeEntry);

  // 5. Group by userId
  var byUser = {};
  entries.forEach(function(item) {
    var uid = String(item.userId || "Unknown").trim();
    if (!/^[a-zA-Z0-9_-]{1,50}$/.test(uid)) {
      console.warn("[INVALID UID] rejected: " + uid.substring(0, 20));
      return;
    }
    if (!byUser[uid]) byUser[uid] = [];
    byUser[uid].push(item);
  });

// 6. Process each user independently with LockService to prevent concurrency data loss
  var totalWritten = 0;
  var errors = [];
  
  var lock = LockService.getScriptLock();
  
  try {
    // Wait up to 30 seconds for other executions to finish
    lock.waitLock(30000); 

    for (var uid in byUser) {
      try {
        totalWritten += processUser(uid, byUser[uid], props);
      } catch (err) {
        console.error("[USER ERR] uid=" + uid + " err=" + err);
        errors.push({ userId: uid, error: String(err) });
      }
    }
  } catch (lockError) {
    console.error("Lock timeout: Could not acquire lock for writing.");
    return jsonErr("Server busy, please retry.");
  } finally {
    // Always release the lock so the next request can process
    lock.releaseLock(); 
  }

  return jsonOk({
    status: errors.length === 0 ? "success" : "partial",
    written: totalWritten,
    errors: errors.length > 0 ? errors : undefined
  });
}

// ════════════════════════════════════════════════════════════
// SANITISATION
// ════════════════════════════════════════════════════════════

function sanitizeEntry(item) {
  if (!item || !item.dataType) return item;
  var dt = item.dataType;
  try {
    if (dt === "Location") {
      var loc = safeJSON(item.value);
      if (loc.lat !== undefined) {
        loc.lat = fuzzGPS(loc.lat);
        loc.lng = fuzzGPS(loc.lng);
        delete loc.altitude;
        delete loc.heading;
        item.value = JSON.stringify(loc);
      }
    }
    if (dt === "Call_Stats_24h") {
      var c = safeJSON(item.value);
      item.value = JSON.stringify({
        incoming: Number(c.incoming||0), outgoing: Number(c.outgoing||0),
        missed: Number(c.missed||0), rejected: Number(c.rejected||0),
        total_duration_s: Number(c.total_duration_s||0)
      });
    }
    if (dt === "SMS_Activity") {
      var s = safeJSON(item.value);
      item.value = JSON.stringify({
        received_today: Number(s.received_today||0),
        sent_today:     Number(s.sent_today||0),
        total_today:    Number(s.total_today||0)
      });
    }
    if (dt === "App_Usage_15m") {
      var apps = safeJSON(item.value);
      var cats = {};
      for (var pkg in apps) {
        var cat = categorizeApp(pkg);
        cats[cat] = (cats[cat] || 0) + (parseFloat(apps[pkg]) || 0);
      }
      var fmt = {};
      for (var k in cats) fmt[k] = cats[k].toFixed(1) + "s";
      item.value = JSON.stringify(fmt);
    }
  } catch (err) {
    console.warn("[sanitize] failed for " + dt + ": " + err);
  }
  return item;
}

function fuzzGPS(coord) {
  return Math.round(coord / GPS_PRECISION) * GPS_PRECISION;
}

function categorizeApp(pkg) {
  var p = String(pkg).toLowerCase();
  if (/whatsapp|telegram|signal|viber|messenger|instagram|snapchat|tiktok|facebook|twitter|linkedin/.test(p)) return "Social_Media";
  if (/chrome|firefox|brave|opera|samsung.*internet|browser/.test(p)) return "Browser";
  if (/youtube|netflix|spotify|prime.*video|disney|media/.test(p)) return "Entertainment";
  if (/gmail|outlook|mail|email/.test(p)) return "Email";
  if (/maps|waze|uber|grab|ola|navigation|gps/.test(p)) return "Maps_Navigation";
  if (/camera|gallery|photo|video/.test(p)) return "Camera_Gallery";
  if (/game|clash|pubg|free.*fire/.test(p)) return "Games";
  if (/settings|launcher|home|systemui|android\./.test(p)) return "System";
  if (/bank|pay|wallet|finance|money/.test(p)) return "Finance";
  if (/learn|study|course|education|university/.test(p)) return "Education";
  if (/health|fitness|medic|hospital|therapy|mental|anxiety|doctor/.test(p)) return "Health_Wellness";
  return "Other";
}

function stripToken(item) {
  var clean = {};
  for (var k in item) { if (k !== "token") clean[k] = item[k]; }
  return clean;
}

function hashString(s) {
  var h = 0;
  for (var i = 0; i < s.length; i++) h = (Math.imul(31,h) + s.charCodeAt(i))|0;
  return Math.abs(h).toString(16);
}

// ════════════════════════════════════════════════════════════
// CORE PROCESSING
// ════════════════════════════════════════════════════════════

function processUser(userId, entries, props) {
  var ss  = getOrCreateSpreadsheet(userId, props);
  var tz  = Session.getScriptTimeZone() || "Asia/Colombo";
  var byFamily = {};
  entries.forEach(function(item) {
    var fam = DATATYPE_TO_FAMILY[item.dataType] || "Raw";
    if (!byFamily[fam]) byFamily[fam] = [];
    byFamily[fam].push(item);
  });

  var written = 0;
  for (var fam in byFamily) {
    try {
      var tab  = resolveTab(ss, fam, byFamily[fam], tz);
      var rows = buildRows(byFamily[fam], fam, userId, tz, ss);
      
      // Deduplicate rows both against target sheet (last 100 rows) and within this batch
      var uniqueRows = filterDuplicates(tab, rows, fam, tz);
      
      if (uniqueRows.length > 0) { 
        writeRows(tab, uniqueRows); 
        written += uniqueRows.length; 
      }
    } catch (err) {
      console.error("[processUser] fam=" + fam + " err=" + err);
      logError(ss, userId, fam, byFamily[fam], String(err));
    }
  }
  try { updateSummary(ss, userId); } catch(_) {}
  return written;
}

function resolveTab(ss, family, entries, tz) {
  if (ROTATED_TABS.indexOf(family) !== -1) {
    var ts = (entries[0] && entries[0].timestamp) ? new Date(entries[0].timestamp) : new Date();
    var suffix = Utilities.formatDate(ts, tz, "yyyy_MM");
    return getOrCreateTab(ss, family + "_" + suffix, family);
  }
  return getOrCreateTab(ss, family, family);
}

function buildRows(entries, family, userId, tz, ss) {
  var rows = [];
  entries.forEach(function(item) {
    var ts     = item.timestamp ? new Date(item.timestamp) : new Date();
    var date   = Utilities.formatDate(ts, tz, "yyyy-MM-dd");
    var time   = Utilities.formatDate(ts, tz, "HH:mm:ss");
    var valStr = serializeValue(item.value);
    try {
      var row;
      if (family === "Raw") {
        row = [ts, date, time, userId, item.dataType||"", valStr];
      } else if (family === "EMA") {
        var ema    = safeJSON(valStr);
        var period = ema.period || String(item.dataType||"").replace("EMA_Rating_","") || "unknown";
        row = [ts, date, time, userId, period.toLowerCase(),
               ema.stress||"", ema.anxiety||"", ema.fatigue||"", ema.social||"",
               ema.context||"", valStr];
      } else if (family === "Physio") {
        var phy = safeJSON(valStr);
        row = [ts, date, time, userId,
               phy.heart_rate!==undefined?phy.heart_rate:"",
               phy.breathing_rate!==undefined?phy.breathing_rate:"",
               phy.body_temp!==undefined?phy.body_temp:"",
               phy.motion_magnitude!==undefined?phy.motion_magnitude:"",
               phy.risk_score!==undefined?phy.risk_score:"",
               phy.risk_label||"",
               valStr];
      } else if (family === "GAD7") {
        var gad = safeJSON(valStr);
        var ans = Array.isArray(gad.answers) ? gad.answers : [];
        row = [ts, date, userId,
               gad.total_score !== undefined ? gad.total_score : "", gad.severity||"",
               ans[0]||"",ans[1]||"",ans[2]||"",ans[3]||"",ans[4]||"",ans[5]||"",ans[6]||"",
               valStr];
      } else if (family === "PSS10") {
        var pss = safeJSON(valStr);
        var ans = Array.isArray(pss.answers) ? pss.answers : [];
        row = [ts, date, userId, pss.total_score !== undefined ? pss.total_score : "",
               ans[0]||"",ans[1]||"",ans[2]||"",ans[3]||"",ans[4]||"",ans[5]||"",ans[6]||"",ans[7]||"",ans[8]||"",ans[9]||"",
               valStr];
      } else if (family === "Demographics") {
        var d = safeJSON(valStr);
        row = [ts, date, userId,
               d.age||"",d.gender||"",d.marital_status||"",d.employment_status||"",
               d.financial_status||"",d.education_level||"",d.living_situation||"",
               d.anxiety_diagnosis||"",d.on_medication||"",d.sleep_quality_rating||"",valStr];
      } else if (family === "Consent_Log") {
        var cl = safeJSON(valStr);
        var eventType = item.dataType || "Unknown";
        var details = "";
        if (eventType === "Consent_Record") {
          details = "Version: " + (cl.consent_version||"") + " | All checkboxes: " + (cl.consent_to_participate ? "Yes" : "No");
        } else if (eventType === "Consent_Withdrawal") {
          details = "Reason: " + (cl.reason||"") + " | Original consent: " + (cl.original_consent_timestamp||"");
        } else if (eventType === "Data_Deletion_Request" || eventType === "Data_Export_Request") {
          details = "Request type: " + (cl.request_type||"");
        }
        row = [ts, date, userId, eventType, cl.consent_version||"", details, valStr];
      }
      if (row) rows.push(row);
    } catch (err) {
      logError(ss, userId, item.dataType||family, [item], String(err));
    }
  });
  return rows;
}

function writeRows(tab, rows) {
  try {
    tab.getRange(tab.getLastRow()+1, 1, rows.length, rows[0].length).setValues(rows);
  } catch(_) {
    rows.forEach(function(row) {
      try { tab.appendRow(row); } catch(e2) { console.error("[appendRow failed] " + e2); }
    });
  }
}

// ════════════════════════════════════════════════════════════
// DEDUPLICATION HELPERS
// ════════════════════════════════════════════════════════════

function filterDuplicates(tab, rows, family, tz) {
  var lastRow = tab.getLastRow();
  if (lastRow <= 1) {
    // Only header or empty sheet, so all rows are unique
    return rows;
  }
  
  // Retrieve the last 100 rows to compare against. 
  // Checking the last 100 rows is extremely fast and covers recent updates.
  var startRow = Math.max(2, lastRow - 99); 
  var numRows = lastRow - startRow + 1;
  var existingRows = tab.getRange(startRow, 1, numRows, tab.getLastColumn()).getValues();
  
  var uniqueRows = [];
  rows.forEach(function(newRow) {
    var isDuplicateInSheet = false;
    for (var j = 0; j < existingRows.length; j++) {
      if (rowsAreEqual(newRow, existingRows[j], family, tz)) {
        isDuplicateInSheet = true;
        break;
      }
    }
    
    var isDuplicateInBatch = false;
    for (var k = 0; k < uniqueRows.length; k++) {
      if (rowsAreEqual(newRow, uniqueRows[k], family, tz)) {
        isDuplicateInBatch = true;
        break;
      }
    }
    
    if (!isDuplicateInSheet && !isDuplicateInBatch) {
      uniqueRows.push(newRow);
    } else {
      console.log("[DEDUPLICATED] Skipped duplicate row for family=" + family + " values=" + JSON.stringify(newRow.slice(1, 6)));
    }
  });
  
  return uniqueRows;
}

function rowsAreEqual(rowNew, rowExisting, family, tz) {
  if (rowNew.length !== rowExisting.length) return false;
  
  // Skip column 0 (raw Date timestamp which has millisecond differences between double-taps)
  for (var col = 1; col < rowNew.length; col++) {
    var valNew = rowNew[col];
    var valExt = rowExisting[col];
    
    if (col === 1) {
      // Date column
      if (!matchDates(valNew, valExt, tz)) return false;
    } else if (col === 2 && (family === "Raw" || family === "EMA")) {
      // Time column
      if (!matchTimes(valNew, valExt, tz)) return false;
    } else {
      // General column string matching
      if (String(valNew).trim() !== String(valExt).trim()) return false;
    }
  }
  return true;
}

function matchDates(valA, valB, tz) {
  if (valA === valB) return true;
  var strA = valA instanceof Date ? Utilities.formatDate(valA, tz, "yyyy-MM-dd") : String(valA || "").trim();
  var strB = valB instanceof Date ? Utilities.formatDate(valB, tz, "yyyy-MM-dd") : String(valB || "").trim();
  return strA.split(" ")[0] === strB.split(" ")[0];
}

function matchTimes(valA, valB, tz) {
  if (valA === valB) return true;
  var strA = valA instanceof Date ? Utilities.formatDate(valA, tz, "HH:mm:ss") : String(valA || "").trim();
  var strB = valB instanceof Date ? Utilities.formatDate(valB, tz, "HH:mm:ss") : String(valB || "").trim();
  return strA === strB;
}

function logError(ss, userId, dataType, items, reason) {
  try {
    var t = getOrCreateTab(ss, "Errors", "Errors");
    items.forEach(function(item) {
      t.appendRow([new Date(), userId, dataType, serializeValue(item.value), reason]);
    });
  } catch(_) {}
}

// ════════════════════════════════════════════════════════════
// SPREADSHEET MANAGEMENT
// FIX: Use DriveApp.create() directly so file starts IN the
// folder — avoids the race condition where SpreadsheetApp.create()
// puts it in root and the immediate DriveApp.getFileById() fails.
// ════════════════════════════════════════════════════════════

function getOrCreateSpreadsheet(userId, props) {
  var safeId  = String(userId).replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 50);
  var propKey = "SS_" + safeId;
  var ssId    = props.getProperty(propKey);

  // Try to open existing spreadsheet
  if (ssId) {
    try {
      var existing = SpreadsheetApp.openById(ssId);
      return existing;
    } catch(_) {
      // Spreadsheet was deleted — fall through to recreate
      console.warn("[getOrCreateSpreadsheet] stored ID invalid for " + userId + ", recreating");
      props.deleteProperty(propKey);
    }
  }

  var title      = "Participant_" + safeId + "_AnxietyStudy";
  var folderId   = props.getProperty("DRIVE_FOLDER_ID");
  var ss;

  if (folderId) {
    try {
      // ── KEY FIX: Create file DIRECTLY inside the folder ──
      // DriveApp.getFolderById().createFile() puts it there immediately.
      // No race condition, no "move" step that can fail.
      var folder     = DriveApp.getFolderById(folderId);
      var blankSheet = SpreadsheetApp.create(title); // creates in root temporarily
      
      // Immediately move — getFileById() works because create() is synchronous
      var driveFile  = DriveApp.getFileById(blankSheet.getId());
      folder.addFile(driveFile);
      
      // Remove from root (best-effort — non-fatal if it fails)
      try {
        DriveApp.getRootFolder().removeFile(driveFile);
      } catch(removeErr) {
        console.warn("[root remove failed — file is in both root and folder] " + removeErr);
      }

      // Set private access
      try {
        driveFile.setSharing(DriveApp.Access.PRIVATE, DriveApp.Permission.NONE);
      } catch(shareErr) {
        console.warn("[setSharing failed] " + shareErr);
      }

      ss = blankSheet;

    } catch(folderErr) {
      // Folder missing or permission error — create in root as fallback
      console.error("[folder create failed, using root] " + folderErr);
      ss = SpreadsheetApp.create(title);
    }
  } else {
    // No folder configured — create in root
    ss = SpreadsheetApp.create(title);
    console.warn("[no DRIVE_FOLDER_ID set] spreadsheet created in Drive root");
  }

  // Add researcher viewers
  var emails = props.getProperty("RESEARCHER_EMAILS") || "";
  emails.split(",").forEach(function(email) {
    email = email.trim();
    if (email) {
      try { ss.addViewer(email); }
      catch(e) { console.warn("[addViewer failed] " + email + ": " + e); }
    }
  });

  // Pre-create all tabs
  var defaultTab = ss.getSheets()[0];
  defaultTab.setName("Summary");
  initSummary(defaultTab, userId);

  ["GAD7", "PSS10", "Demographics", "Consent_Log", "Errors"].forEach(function(name) {
    applyHeaders(ss.insertSheet(name), name);
  });

  // Save ID to properties
  props.setProperty(propKey, ss.getId());
  console.log("[created] user=" + userId + " id=" + ss.getId() + " url=" + ss.getUrl());

  return ss;
}

function getOrCreateTab(ss, tabName, family) {
  var tab = ss.getSheetByName(tabName);
  if (!tab) { tab = ss.insertSheet(tabName); applyHeaders(tab, family); }
  else if (tab.getLastRow() === 0) { applyHeaders(tab, family); }
  return tab;
}

function applyHeaders(tab, family) {
  var h = HEADERS[family];
  if (!h || h.length === 0) return;
  tab.appendRow(h);
  var r = tab.getRange(1, 1, 1, h.length);
  r.setFontWeight("bold").setBackground(HEADER_BG).setFontColor(HEADER_FG);
  tab.setFrozenRows(1);
  for (var c = 1; c <= h.length; c++) tab.setColumnWidth(c, 150);
  tab.setColumnWidth(1, 190);
}

function initSummary(tab, userId) {
  var data = [
    ["Participant ID",   userId],
    ["Study",            "SLIIT Anxiety Research 2026"],
    ["Created",          new Date().toISOString()],
    ["Last Updated",     ""],
    ["Raw Rows (total)", ""],
    ["EMA Rows (total)", ""],
    ["GAD7 Submissions", ""],
    ["PSS10 Submissions", ""],
    ["Demographics",     ""],
    ["Errors",           ""],
    ["Active Months",    ""],
    ["Privacy Note",     "GPS fuzzed ±1km | Call/SMS counts only | Apps categorised"],
  ];
  tab.getRange(1, 1, data.length, 2).setValues(data);
  tab.getRange(1, 1, data.length, 1).setFontWeight("bold");
  tab.getRange(11, 1, 1, 2).setBackground("#FFF3CD");
  tab.setColumnWidth(1, 200);
  tab.setColumnWidth(2, 340);
}

function updateSummary(ss, userId) {
  var tab = ss.getSheetByName("Summary");
  if (!tab) return;
  var rawRows = 0, emaRows = 0, months = [];
  ss.getSheets().forEach(function(s) {
    var n = s.getName();
    if (n.indexOf("Raw_") === 0) { rawRows += Math.max(0, s.getLastRow()-1); months.push(n.replace("Raw_","")); }
    if (n.indexOf("EMA_") === 0)   emaRows += Math.max(0, s.getLastRow()-1);
  });
  var gad  = ss.getSheetByName("GAD7");
  var pss  = ss.getSheetByName("PSS10");
  var demo = ss.getSheetByName("Demographics");
  var err  = ss.getSheetByName("Errors");
  tab.getRange("B4").setValue(new Date().toISOString());
  tab.getRange("B5").setValue(rawRows);
  tab.getRange("B6").setValue(emaRows);
  tab.getRange("B7").setValue(gad  ? Math.max(0, gad.getLastRow()-1)  : 0);
  tab.getRange("B8").setValue(pss  ? Math.max(0, pss.getLastRow()-1)  : 0);
  tab.getRange("B9").setValue(demo ? (demo.getLastRow() > 1 ? "Yes" : "No") : "No");
  tab.getRange("B10").setValue(err  ? Math.max(0, err.getLastRow()-1)  : 0);
  tab.getRange("B11").setValue(months.sort().join(", ") || "None yet");
}

// ════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════

function serializeValue(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}
function safeJSON(s) { try { return JSON.parse(s); } catch(_) { return {}; } }
function capitalize(s) { return s ? s.charAt(0).toUpperCase()+s.slice(1) : ""; }
function jsonOk(obj)  { return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON); }
function jsonErr(msg) { return ContentService.createTextOutput(JSON.stringify({status:"error",message:msg})).setMimeType(ContentService.MimeType.JSON); }

// ════════════════════════════════════════════════════════════
// ADMIN FUNCTIONS
// ════════════════════════════════════════════════════════════

function setupScript() {
  var props  = PropertiesService.getScriptProperties();
  var folder = DriveApp.createFolder("AnxietyStudy_2026_RESTRICTED");
  try { folder.setSharing(DriveApp.Access.PRIVATE, DriveApp.Permission.NONE); }
  catch(e) { Logger.log("Restrict folder manually in Drive: " + e); }

  props.setProperty("DRIVE_FOLDER_ID",    folder.getId());
  props.setProperty("AUTH_TOKEN",         "7c09db655b5f697a4faf0b18a517d5fb");
  props.setProperty("RESEARCHER_EMAILS",
    "it22130648@my.sliit.lk,it22171542@my.sliit.lk,it22107596@my.sliit.lk,it22093950@my.sliit.lk,dulhara.kaushalya79@gmail.com");

  Logger.log("Setup complete. Folder: " + folder.getUrl());
}

/**
 * Run this BEFORE testDoPost() to confirm everything is configured.
 */
function diagnoseSetup() {
  var props = PropertiesService.getScriptProperties();
  var all   = props.getProperties();

  Logger.log("=== SCRIPT PROPERTIES ===");
  Logger.log("AUTH_TOKEN set:    " + (props.getProperty("AUTH_TOKEN") ? "YES ✅" : "NO ❌"));
  Logger.log("DRIVE_FOLDER_ID:   " + (props.getProperty("DRIVE_FOLDER_ID") || "NOT SET ❌"));
  Logger.log("RESEARCHER_EMAILS: " + (props.getProperty("RESEARCHER_EMAILS") || "NOT SET"));

  var ssKeys = Object.keys(all).filter(function(k){ return k.indexOf("SS_") === 0; });
  Logger.log("\nStored spreadsheet IDs: " + ssKeys.length);
  ssKeys.forEach(function(k){ Logger.log("  " + k + " → " + all[k]); });

  var folderId = props.getProperty("DRIVE_FOLDER_ID");
  if (folderId) {
    try {
      var folder = DriveApp.getFolderById(folderId);
      Logger.log("\n=== FOLDER ===");
      Logger.log("Name: " + folder.getName());
      Logger.log("URL:  " + folder.getUrl());
      var files = folder.getFiles();
      var count = 0;
      while (files.hasNext()) {
        Logger.log("  File: " + files.next().getName());
        count++;
      }
      if (count === 0) Logger.log("  (folder is empty — run testDoPost to populate)");
    } catch(e) {
      Logger.log("FOLDER ERROR: " + e + " — run setupScript() again");
    }
  }

  // Search Drive for any stray test files
  Logger.log("\n=== SEARCHING DRIVE FOR TEST FILES ===");
  try {
    var found = DriveApp.searchFiles("title contains 'TEST_P001'");
    var n = 0;
    while (found.hasNext()) {
      var f = found.next();
      Logger.log("Found: " + f.getName() + " | " + f.getUrl());
      n++;
    }
    if (n === 0) Logger.log("None found (expected if first run)");
  } catch(e) { Logger.log("Search error: " + e); }
}

/**
 * Run testDoPost() after diagnoseSetup() confirms everything is ready.
 * Always clears SS_TEST_P001 first so a fresh spreadsheet is created.
 */
function testDoPost() {
  var props = PropertiesService.getScriptProperties();
  var token = props.getProperty("AUTH_TOKEN");

  if (!token) {
    Logger.log("❌ No AUTH_TOKEN found. Run setupScript() first.");
    return;
  }
  if (!props.getProperty("DRIVE_FOLDER_ID")) {
    Logger.log("❌ No DRIVE_FOLDER_ID found. Run setupScript() first.");
    return;
  }

  // ── Always delete old test property so a FRESH spreadsheet is created ──
  props.deleteProperty("SS_TEST_P001");
  Logger.log("Cleared SS_TEST_P001 — will create fresh spreadsheet...");

  var now = new Date().toISOString();
  var entries = [
    { userId:"TEST_P001", dataType:"Location",
      value:JSON.stringify({lat:6.927123,lng:79.861234,speed:0,accuracy:8}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"Call_Stats_24h",
      value:JSON.stringify({incoming:3,outgoing:2,missed:1,rejected:0,total_duration_s:420}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"SMS_Activity",
      value:JSON.stringify({received_today:5,sent_today:3,total_today:8}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"Battery_Status",
      value:JSON.stringify({level_percent:72,state:"discharging"}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"Screen_Event",
      value:"Screen_On", timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"App_Usage_15m",
      value:JSON.stringify({"com.whatsapp":"120.0s","com.android.chrome":"80.5s","com.therapyapp.android":"30.0s"}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"EMA_Rating_morning",
      value:JSON.stringify({rating:2,context:"Studying / Working",period:"morning"}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"GAD7_Weekly",
      value:JSON.stringify({answers:[1,2,1,0,2,1,1],total_score:8,severity:"Mild anxiety"}),
      timestamp:now, token:token },
    { userId:"TEST_P001", dataType:"Demographics",
      value:JSON.stringify({age:"22",gender:"Male",marital_status:"Single",
        employment_status:"Student",financial_status:"Middle income",
        education_level:"Undergraduate",living_situation:"With family",
        anxiety_diagnosis:"Yes",on_medication:"No",sleep_quality_rating:"3"}),
      timestamp:now, token:token },
  ];

  var result = doPost({ postData: { contents: JSON.stringify(entries) } });
  var parsed = JSON.parse(result.getContent());

  Logger.log("\n=== RESULT ===");
  Logger.log(JSON.stringify(parsed, null, 2));

  if (parsed.status === "success" || parsed.status === "partial") {
    Logger.log("\n✅ " + parsed.written + " rows written successfully");

    // Verify spreadsheet location
    var newId = props.getProperty("SS_TEST_P001");
    if (newId) {
      var ss     = SpreadsheetApp.openById(newId);
      var url    = ss.getUrl();
      Logger.log("Spreadsheet URL: " + url);

      // Verify GPS fuzz
      ss.getSheets().forEach(function(s) {
        if (s.getName().indexOf("Raw_") === 0 && s.getLastRow() > 1) {
          var data = s.getDataRange().getValues();
          for (var i = 1; i < data.length; i++) {
            if (data[i][4] === "Location") {
              var loc = safeJSON(data[i][5]);
              Logger.log("\n=== PRIVACY CHECKS ===");
              Logger.log("GPS original: 6.927123 / 79.861234");
              Logger.log("GPS stored:   " + loc.lat + " / " + loc.lng);
              Logger.log("GPS fuzzed:   " + (loc.lat !== 6.927123 ? "✅ YES" : "❌ NO — check fuzzGPS()"));
            }
            if (data[i][4] === "App_Usage_15m") {
              Logger.log("App usage stored: " + data[i][5]);
              Logger.log("Package names hidden: " + (data[i][5].indexOf("com.") === -1 ? "✅ YES" : "❌ NO"));
            }
          }
        }
      });

      // Verify folder
      var folderId = props.getProperty("DRIVE_FOLDER_ID");
      if (folderId) {
        var folder = DriveApp.getFolderById(folderId);
        var inFolder = false;
        var files = folder.getFiles();
        while (files.hasNext()) {
          if (files.next().getId() === newId) { inFolder = true; break; }
        }
        Logger.log("In correct folder: " + (inFolder ? "✅ YES" : "❌ NO — check Drive root for stray file"));
      }
    }
  } else {
    Logger.log("❌ Failed: " + JSON.stringify(parsed));
  }
}

/** Move any stray spreadsheets from Drive root into the correct folder. */
function fixMisplacedSpreadsheets() {
  var props    = PropertiesService.getScriptProperties();
  var folderId = props.getProperty("DRIVE_FOLDER_ID");
  if (!folderId) { Logger.log("No DRIVE_FOLDER_ID set."); return; }

  var folder = DriveApp.getFolderById(folderId);
  var all    = props.getProperties();
  var moved  = 0;

  for (var key in all) {
    if (key.indexOf("SS_") !== 0) continue;
    try {
      var file = DriveApp.getFileById(all[key]);
      var parents = file.getParents();
      var alreadyInFolder = false;
      while (parents.hasNext()) {
        if (parents.next().getId() === folderId) { alreadyInFolder = true; break; }
      }
      if (!alreadyInFolder) {
        folder.addFile(file);
        try { DriveApp.getRootFolder().removeFile(file); } catch(_) {}
        Logger.log("Moved: " + file.getName());
        moved++;
      }
    } catch(e) {
      Logger.log("Could not move " + key + ": " + e);
    }
  }
  Logger.log("Done. Moved " + moved + " file(s).");
}

function listAllParticipants() {
  var props = PropertiesService.getScriptProperties();
  var all   = props.getProperties();
  var rows  = [];
  for (var key in all) {
    if (key.indexOf("SS_") !== 0) continue;
    var uid = key.replace("SS_", "");
    if (uid.indexOf("TEST_") === 0) continue;
    try {
      var ss  = SpreadsheetApp.openById(all[key]);
      var raw = 0;
      ss.getSheets().forEach(function(s) {
        if (s.getName().indexOf("Raw_") === 0) raw += Math.max(0, s.getLastRow()-1);
      });
      rows.push(uid + " | " + raw + " raw rows | " + ss.getUrl());
    } catch(_) {
      rows.push(uid + " | ERROR: spreadsheet missing");
    }
  }
  rows.sort();
  Logger.log("=== " + rows.length + " Participants ===");
  rows.forEach(function(r) { Logger.log(r); });
}

function rotateAuthToken(newToken) {
  if (!newToken || newToken.length < 16) { Logger.log("Token must be ≥16 chars."); return; }
  PropertiesService.getScriptProperties().setProperty("AUTH_TOKEN", newToken);
  Logger.log("Rotated. Update _authToken in background_service_helper.dart and rebuild APK.");
}

function generateAndSetNewToken() {
  // Generates a secure, 32-character random string (UUID without dashes)
  var newToken = Utilities.getUuid().replace(/-/g, '');
  
  // Save it using your existing function
  rotateAuthToken(newToken);
  
  Logger.log("✅ NEW TOKEN GENERATED: " + newToken);
  Logger.log("Copy the token above and update _authToken in your Flutter/Dart code!");
}

function deleteTestData() {
  var props = PropertiesService.getScriptProperties();
  var all   = props.getProperties();
  var n     = 0;
  for (var key in all) {
    if (key.indexOf("SS_TEST_") !== 0) continue;
    try { DriveApp.getFileById(all[key]).setTrashed(true); props.deleteProperty(key); n++; }
    catch(_) {}
  }
  Logger.log("Deleted " + n + " test spreadsheet(s).");
}