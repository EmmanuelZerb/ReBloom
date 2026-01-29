/**
 * Signed URL Generator
 * Generates and validates signed URLs for secure file access
 * Prevents unauthorized access through URL guessing/brute-forcing
 */

import { createHmac, timingSafeEqual } from 'crypto';
import { config_ } from './config';
import { apiLogger as logger } from './logger';

// ============================================
// Configuration
// ============================================

// Use WEBHOOK_SECRET as signing key, or generate a random one
const SIGNING_KEY = config_.webhook.secret || process.env.SIGNING_KEY || generateFallbackKey();
const DEFAULT_EXPIRY_SECONDS = 24 * 60 * 60; // 24 hours

function generateFallbackKey(): string {
  // In production, this should be set via environment variable
  if (config_.isProd) {
    logger.warn('No SIGNING_KEY or WEBHOOK_SECRET configured - using random key (URLs will not persist across restarts)');
  }
  return require('crypto').randomBytes(32).toString('hex');
}

// ============================================
// Signed URL Generation
// ============================================

interface SignedUrlParams {
  /** The file path or resource ID */
  path: string;
  /** The job ID associated with this file */
  jobId: string;
  /** Expiry time in seconds (default: 24 hours) */
  expiresIn?: number;
}

interface SignedUrlComponents {
  path: string;
  jobId: string;
  expires: number;
  signature: string;
}

/**
 * Generate a signed URL for file access
 */
export function generateSignedUrl(params: SignedUrlParams): string {
  const { path, jobId, expiresIn = DEFAULT_EXPIRY_SECONDS } = params;

  const expires = Math.floor(Date.now() / 1000) + expiresIn;
  const dataToSign = `${path}:${jobId}:${expires}`;

  const signature = createHmac('sha256', SIGNING_KEY)
    .update(dataToSign)
    .digest('hex')
    .substring(0, 32); // Truncate for shorter URLs

  // Encode components for URL
  const token = Buffer.from(
    JSON.stringify({ path, jobId, expires, signature })
  ).toString('base64url');

  return `/api/download/${jobId}?token=${token}`;
}

/**
 * Parse and validate a signed URL token
 */
export function validateSignedUrl(
  token: string,
  expectedJobId: string
): { valid: boolean; error?: string; path?: string } {
  try {
    // Decode token
    const decoded = Buffer.from(token, 'base64url').toString('utf-8');
    const components: SignedUrlComponents = JSON.parse(decoded);

    const { path, jobId, expires, signature } = components;

    // Check job ID matches
    if (jobId !== expectedJobId) {
      logger.warn({ expectedJobId, actualJobId: jobId }, 'Job ID mismatch in signed URL');
      return { valid: false, error: 'Invalid token' };
    }

    // Check expiry
    const now = Math.floor(Date.now() / 1000);
    if (now > expires) {
      logger.debug({ jobId, expires, now }, 'Signed URL expired');
      return { valid: false, error: 'URL has expired' };
    }

    // Verify signature
    const dataToSign = `${path}:${jobId}:${expires}`;
    const expectedSignature = createHmac('sha256', SIGNING_KEY)
      .update(dataToSign)
      .digest('hex')
      .substring(0, 32);

    const providedBuffer = Buffer.from(signature, 'hex');
    const expectedBuffer = Buffer.from(expectedSignature, 'hex');

    if (providedBuffer.length !== expectedBuffer.length) {
      return { valid: false, error: 'Invalid signature' };
    }

    if (!timingSafeEqual(providedBuffer, expectedBuffer)) {
      logger.warn({ jobId }, 'Invalid signature in signed URL');
      return { valid: false, error: 'Invalid signature' };
    }

    return { valid: true, path };
  } catch (error) {
    logger.warn({ error }, 'Failed to parse signed URL token');
    return { valid: false, error: 'Invalid token format' };
  }
}

/**
 * Generate a simple download token for a job
 * This is a simpler version that just validates ownership
 */
export function generateDownloadToken(jobId: string, expiresIn: number = DEFAULT_EXPIRY_SECONDS): string {
  const expires = Math.floor(Date.now() / 1000) + expiresIn;
  const dataToSign = `download:${jobId}:${expires}`;

  const signature = createHmac('sha256', SIGNING_KEY)
    .update(dataToSign)
    .digest('hex')
    .substring(0, 16);

  return `${expires}.${signature}`;
}

/**
 * Validate a simple download token
 */
export function validateDownloadToken(token: string, jobId: string): boolean {
  try {
    const [expiresStr, signature] = token.split('.');
    const expires = parseInt(expiresStr, 10);

    if (isNaN(expires)) {
      return false;
    }

    // Check expiry
    const now = Math.floor(Date.now() / 1000);
    if (now > expires) {
      return false;
    }

    // Verify signature
    const dataToSign = `download:${jobId}:${expires}`;
    const expectedSignature = createHmac('sha256', SIGNING_KEY)
      .update(dataToSign)
      .digest('hex')
      .substring(0, 16);

    const providedBuffer = Buffer.from(signature, 'hex');
    const expectedBuffer = Buffer.from(expectedSignature, 'hex');

    if (providedBuffer.length !== expectedBuffer.length) {
      return false;
    }

    return timingSafeEqual(providedBuffer, expectedBuffer);
  } catch {
    return false;
  }
}
