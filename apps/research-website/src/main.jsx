import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { normalizeDirectPath } from "./lib/directPath.js";
import "./styles/site.css";
import "./styles/pages.css";

// Normalize direct path URLs before HashRouter reads browser location.
normalizeDirectPath(window.location, window.history);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
