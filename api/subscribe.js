const twilio = require('twilio');
const { createClient } = require('@supabase/supabase-js');

const twilioClient = twilio(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

async function sendSMS(to, body) {
  await twilioClient.messages.create({
    to,
    from: process.env.TWILIO_FROM_NUMBER,
    body,
  });
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const {
    action,
    phone,
    receive_window_threshold,
    receive_quiet,
    receive_digest,
    digest_hour,
    windows,
    is_update,
  } = req.body || {};

  if (!phone) {
    return res.status(400).json({ error: 'phone is required' });
  }

  try {
    // ── Unsubscribe ─────────────────────────────────────────────
    if (action === 'unsubscribe') {
      await supabase
        .from('subscribers')
        .update({ is_active: false })
        .eq('phone_number', phone);

      await sendSMS(phone, "You've been unsubscribed from RSF alerts.");
      return res.status(200).json({ ok: true });
    }

    // ── Subscribe / update preferences ──────────────────────────
    const rwt = receive_window_threshold ?? true;
    const rq  = receive_quiet ?? true;
    const rd  = receive_digest ?? true;
    const dh  = digest_hour ?? 6;

    await supabase.from('subscribers').upsert(
      {
        phone_number: phone,
        receive_window_threshold: rwt,
        receive_quiet: rq,
        receive_digest: rd,
        digest_hour: dh,
        is_active: true,
      },
      { onConflict: 'phone_number' }
    );

    // Replace windows: delete existing, then insert new ones
    await supabase
      .from('subscriber_windows')
      .delete()
      .eq('phone_number', phone);

    if (Array.isArray(windows) && windows.length > 0) {
      await supabase
        .from('subscriber_windows')
        .insert(windows.map(w => ({ ...w, phone_number: phone })));
    }

    // Confirmation SMS
    if (is_update) {
      await sendSMS(phone, 'Your RSF alert preferences have been updated!');
    } else {
      const enabled = [];
      if (rwt) enabled.push('Window Threshold');
      if (rq)  enabled.push('Quieter Than Usual');
      if (rd)  enabled.push('Daily Digest');
      const list = enabled.map(e => `\u2022 ${e}`).join('\n');
      await sendSMS(
        phone,
        `You're subscribed to RSF Weight Room alerts! You'll receive:\n${list}\nReply STOP to unsubscribe anytime.`
      );
    }

    return res.status(200).json({ ok: true });
  } catch (err) {
    console.error('[subscribe]', err);
    return res.status(500).json({ error: err.message });
  }
};
