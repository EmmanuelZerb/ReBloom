/**
 * Webhook Signature Validation
 * Validates incoming webhooks from Replicate using HMAC-SHA256
 */

import { createHmac, timingSafeEqual } from 'crypto';
import { Context, Next } from 'hono';
import { config_ } from './config';
import { apiLogger as logger } from './logger';
import { WebhookSignatureError } from './errors';

// ============================================
// HMAC Signature Verification
// ============================================

/**
 * Verifies a webhook signature using HMAC-SHA256
 * @param payload - The raw request body as a string
 * @param signature - The signature from the webhook header
 * @param secret - The webhook secret
 * @returns true if signature is valid
 */
export function verifyWebhookSignature(
  payload: string,
  signature: string,
  secret: string
): boolean {
  if (!signature || !secret) {
    return false;
  }

  try {
    // Replicate uses the format: sha256=<hex signature>
    const [algorithm, providedHash] = signature.split('=');

    if (algorithm !== 'sha256' || !providedHash) {
      return false;
    }

    const expectedHash = createHmac('sha256', secret)
      .update(payload, 'utf8')
      .digest('hex');

    // Use timing-safe comparison to prevent timing attacks
    const providedBuffer = Buffer.from(providedHash, 'hex');
    const expectedBuffer = Buffer.from(expectedHash, 'hex');

    if (providedBuffer.length !== expectedBuffer.length) {
      return false;
    }

    return timingSafeEqual(providedBuffer, expectedBuffer);
  } catch (error) {
    logger.error({ error }, 'Error verifying webhook signature');
    return false;
  }
}

// ============================================
// Webhook Validation Middleware
// ============================================

/**
 * Middleware to validate Replicate webhook signatures
 * Requires WEBHOOK_SECRET to be configured in environment
 */
export function validateWebhookSignature() {
  return async (c: Context, next: Next) => {
    const webhookSecret = config_.webhook.secret;

    // If no secret configured, skip validation in dev mode (with warning)
    if (!webhookSecret) {
      if (config_.isDev) {
        logger.warn('Webhook secret not configured - skipping signature validation in dev mode');
        await next();
        return;
      } else {
        logger.error('Webhook secret not configured in production');
        throw new WebhookSignatureError({ reason: 'Webhook secret not configured' });
      }
    }

    // Get signature from header
    // Replicate uses 'webhook-signature' or 'x-webhook-signature' header
    const signature =
      c.req.header('webhook-signature') ||
      c.req.header('x-webhook-signature') ||
      c.req.header('x-replicate-signature');

    if (!signature) {
      logger.warn('Webhook request missing signature header');
      throw new WebhookSignatureError({ reason: 'Missing signature header' });
    }

    // Get raw body for signature verification
    // We need to read the body as text to verify the signature
    const rawBody = await c.req.text();

    // Verify signature
    const isValid = verifyWebhookSignature(rawBody, signature, webhookSecret);

    if (!isValid) {
      logger.warn({ signature: signature.substring(0, 20) + '...' }, 'Invalid webhook signature');
      throw new WebhookSignatureError({ reason: 'Signature verification failed' });
    }

    logger.debug('Webhook signature verified successfully');

    // Parse the body as JSON and store it for the handler
    try {
      const jsonBody = JSON.parse(rawBody);
      c.set('webhookPayload', jsonBody);
    } catch (error) {
      logger.error({ error }, 'Failed to parse webhook body as JSON');
      throw new WebhookSignatureError({ reason: 'Invalid JSON payload' });
    }

    await next();
  };
}

/**
 * Generate a webhook signature for testing purposes
 */
export function generateWebhookSignature(payload: string, secret: string): string {
  const hash = createHmac('sha256', secret)
    .update(payload, 'utf8')
    .digest('hex');
  return `sha256=${hash}`;
}
