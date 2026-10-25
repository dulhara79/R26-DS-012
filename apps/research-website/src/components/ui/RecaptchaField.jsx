import { useEffect, useRef, useState } from "react";

// The script may load after React StrictMode mounts and unmounts this component.
// One shared loader, with Google's explicit onload callback, prevents event races.
let captchaApiPromise;

function loadCaptchaApi() {
  if (window.grecaptcha?.render) {
    return Promise.resolve(window.grecaptcha);
  }
  if (!captchaApiPromise) {
    captchaApiPromise = new Promise((resolve, reject) => {
      const callbackName = "__r26ResearchCaptchaReady";
      const script = document.createElement("script");

      const fail = () => {
        delete window[callbackName];
        script.remove();
        captchaApiPromise = undefined;
        reject(new Error("Google reCAPTCHA could not load."));
      };

      // Google's callback runs after the reCAPTCHA rendering API is ready.
      window[callbackName] = () => {
        if (!window.grecaptcha?.render) {
          fail();
          return;
        }
        delete window[callbackName];
        resolve(window.grecaptcha);
      };

      script.src =
        "https://www.google.com/recaptcha/api.js?onload=" +
        callbackName +
        "&render=explicit";
      script.async = true;
      script.defer = true;
      script.onerror = fail;
      document.head.appendChild(script);
    });
  }
  return captchaApiPromise;
}

// EmailJS requires reCAPTCHA v2. The secret key stays in the EmailJS dashboard.
export default function RecaptchaField({ siteKey, onToken, resetCounter }) {
  const containerRef = useRef(null);
  const widgetIdRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    if (!siteKey || !containerRef.current) return;

    let active = true;
    setStatus("loading");

    // A fresh mount point avoids the "already rendered" error when StrictMode
    // replays effects or a visitor navigates away and returns.
    const mountPoint = document.createElement("div");
    containerRef.current.appendChild(mountPoint);

    loadCaptchaApi()
      .then((api) => {
        if (!active) return;
        try {
          widgetIdRef.current = api.render(mountPoint, {
            sitekey: siteKey,
            callback: (value) => {
              if (active) onToken(value);
            },
            "expired-callback": () => {
              if (active) onToken("");
            },
            "error-callback": () => {
              if (active) {
                onToken("");
                setStatus("failed");
              }
            },
          });
          setStatus("ready");
        } catch (error) {
          console.error("Unable to render reCAPTCHA:", error);
          setStatus("failed");
        }
      })
      .catch((error) => {
        if (!active) return;
        console.error("Unable to load reCAPTCHA:", error);
        setStatus("failed");
      });

    return () => {
      active = false;
      widgetIdRef.current = null;
      mountPoint.remove();
    };
  }, [siteKey, onToken, retryCount]);

  useEffect(() => {
    if (resetCounter > 0 && widgetIdRef.current !== null && window.grecaptcha?.reset) {
      window.grecaptcha.reset(widgetIdRef.current);
    }
  }, [resetCounter]);

  if (!siteKey) return null;

  return (
    <div className="contact-turnstile">
      <p className="contact-security-label">Security verification</p>
      <div ref={containerRef} />
      {status === "loading" && (
        <p className="contact-captcha-status" role="status">
          Loading security verification…
        </p>
      )}
      {status === "failed" && (
        <div className="contact-captcha-error" role="alert">
          <p>
            Verification could not load. Check that this site's hostname is
            allowed in your reCAPTCHA settings and that your browser is not
            blocking Google reCAPTCHA.
          </p>
          <button
            className="contact-captcha-retry"
            type="button"
            onClick={() => setRetryCount((value) => value + 1)}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
