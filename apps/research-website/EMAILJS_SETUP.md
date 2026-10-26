# EmailJS — component-to-researcher contact routing

The Contact Us form sends enquiries from the visitor's browser through EmailJS.
One connected sending account is used, and the message is directed to **the selected researcher's individual inbox**, not a shared destination.
No custom backend or Vercel Function is needed.

## 1. Connect one sending account

Create an EmailJS account at https://www.emailjs.com/ and link a sending email service under Email Services.
A dedicated project sending identity is recommended; only that sending account needs to connect.
The four research team members do not connect their personal mailboxes — they receive emails normally.

## 2. One EmailJS template for four fixed recipients

Create one template (the free plan includes up to two). **Do not use a browser-supplied dynamic To Email field such as {{to_email}}.**

In the **To Email** field, paste this exact *single-line* expression (there must be no newlines or separators):

{{#to_c1}}hdsendanayake01@gmail.com{{/to_c1}}{{#to_c2}}senuvilayathma20@gmail.com{{/to_c2}}{{#to_c3}}dulhara.kaushalya79@gmail.com{{/to_c3}}{{#to_c4}}uvisends24946@gmail.com{{/to_c4}}

These existing, publicly listed research team emails correspond to:
- C1 — Sendanayake H.D. (physiological monitoring)
- C2 — Layathma B.M.A.S. (behavioural analysis)
- C3 — Kaushalya I.G.D. (clinical NLP)
- C4 — Seneviratne K.A.U.A. (fusion and evidence)

Use the following other EmailJS template fields:

- Subject: [R26-DS-012] {{component}} — {{subject}}
- Reply-To: {{reply_to}}
- From Name: R26-DS-012 Research Website (the sender email comes from the connected service)
- Body:

New research website enquiry

Research component: {{component}}
Researcher: {{researcher}}

Visitor: {{from_name}}
Visitor email: {{reply_to}}

Subject: {{subject}}

Message:
{{message}}

The frontend passes exactly one true to_c1/to_c2/to_c3/to_c4 Boolean. Conditional Mustache blocks keep the destination addresses fixed in the dashboard template. This prevents a visitor from supplying a different arbitrary To address as a template parameter, but does not replace anti-spam controls. Verify the To Email expression using the EmailJS template tester for each component; use one real address per test.

## 3. Enable mandatory CAPTCHA

EmailJS supports **Google reCAPTCHA v2**, not Cloudflare Turnstile, for server-verified template CAPTCHA.

1. Create a reCAPTCHA v2 checkbox widget for your actual website hostnames at https://www.google.com/recaptcha/admin.
2. Under EmailJS template Settings, enable **reCAPTCHA v2 verification** and store the reCAPTCHA secret **in the EmailJS dashboard only**.
3. Use the public site key in Vite's VITE_RECAPTCHA_SITE_KEY configuration.
4. The website sends the solved token as g-recaptcha-response with the template parameters.
5. Test rejection without CAPTCHA: a request missing the token should be rejected by EmailJS.

## 4. Configure public build variables

Set these build-time environment variables in your website host:

VITE_EMAILJS_SERVICE_ID = service ID from EmailJS Email Services
VITE_EMAILJS_TEMPLATE_ID = template ID from EmailJS Email Templates
VITE_EMAILJS_PUBLIC_KEY = public key from EmailJS Account
VITE_RECAPTCHA_SITE_KEY = public Google reCAPTCHA v2 site key

All four identifiers are public-facing by design. Do **not** include your linked mail account's credentials, EmailJS private key, or Google reCAPTCHA secret in Vite variables or GitHub.
Redeploy after changing any VITE_ variable.

The form deliberately disables submission and tells the visitor to use listed email addresses until all configuration is present.

## 5. Verify end-to-end

Use EmailJS's Test It feature to verify each of four To Email cases, then send through the deployed Contact Us form to C1, C2, C3 and C4. Confirm:
- Each submission is received in the selected researcher's own mailbox; no other researcher's inbox gets a copy.
- The email has a sensible subject and Reply-To points to the visitor's email.
- Invalid input, failed CAPTCHA and unavailable EmailJS produce a visible error, not fake confirmation.
- EmailJS accepting a send request is **not** a guarantee of delivery; check spam folders and provider logs as needed.

Free plans have request quotas; check current limits at https://www.emailjs.com/pricing/.

### Safety notes
Do not collect patient notes, patient names, participant records or other confidential clinical material through this general contact form.
Configure EmailJS reCAPTCHA verification and rate limits; browser-side recipient choices and input validation alone are not security controls.
With the fixed To Email template expression above, the destination emails are not independently provided by visitors, even if they modify the browser form.

### No backend requirement
The previous Resend/Vercel function and Cloudflare Turnstile integration are removed by this change.
Static site deployment is supported, as long as the browser can make requests to EmailJS and the reCAPTCHA widget is configured.
