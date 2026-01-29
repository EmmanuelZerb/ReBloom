/**
 * Tests for signed URL utilities
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  generateDownloadToken,
  validateDownloadToken,
  generateSignedUrl,
  validateSignedUrl,
} from '../src/lib/signed-urls';

describe('Signed URLs', () => {
  describe('generateDownloadToken / validateDownloadToken', () => {
    it('should generate and validate a valid token', () => {
      const jobId = 'test-job-123';
      const token = generateDownloadToken(jobId);

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(validateDownloadToken(token, jobId)).toBe(true);
    });

    it('should reject tokens with wrong job ID', () => {
      const jobId = 'test-job-123';
      const wrongJobId = 'test-job-456';
      const token = generateDownloadToken(jobId);

      expect(validateDownloadToken(token, wrongJobId)).toBe(false);
    });

    it('should reject expired tokens', () => {
      const jobId = 'test-job-123';
      // Generate token that expires in -1 second (already expired)
      const token = generateDownloadToken(jobId, -1);

      expect(validateDownloadToken(token, jobId)).toBe(false);
    });

    it('should reject malformed tokens', () => {
      const jobId = 'test-job-123';

      expect(validateDownloadToken('', jobId)).toBe(false);
      expect(validateDownloadToken('invalid', jobId)).toBe(false);
      expect(validateDownloadToken('not.valid.token', jobId)).toBe(false);
      expect(validateDownloadToken('12345.invalidhex', jobId)).toBe(false);
    });

    it('should reject tampered tokens', () => {
      const jobId = 'test-job-123';
      const token = generateDownloadToken(jobId);

      // Tamper with expiry
      const [expiry, sig] = token.split('.');
      const tamperedToken = `${parseInt(expiry) + 1000}.${sig}`;

      expect(validateDownloadToken(tamperedToken, jobId)).toBe(false);
    });

    it('should respect custom expiry times', () => {
      const jobId = 'test-job-123';
      const shortExpiry = 10; // 10 seconds
      const token = generateDownloadToken(jobId, shortExpiry);

      // Token should be valid immediately
      expect(validateDownloadToken(token, jobId)).toBe(true);

      // Parse expiry from token
      const [expiryStr] = token.split('.');
      const expiry = parseInt(expiryStr, 10);
      const now = Math.floor(Date.now() / 1000);

      // Expiry should be approximately now + shortExpiry
      expect(expiry).toBeGreaterThanOrEqual(now);
      expect(expiry).toBeLessThanOrEqual(now + shortExpiry + 1);
    });
  });

  describe('generateSignedUrl / validateSignedUrl', () => {
    it('should generate and validate a signed URL', () => {
      const path = 'processed/abc123.png';
      const jobId = 'test-job-123';

      const url = generateSignedUrl({ path, jobId });

      expect(url).toContain('/api/download/');
      expect(url).toContain('token=');

      // Extract token from URL
      const token = url.split('token=')[1];
      const result = validateSignedUrl(token, jobId);

      expect(result.valid).toBe(true);
      expect(result.path).toBe(path);
    });

    it('should reject URLs with wrong job ID', () => {
      const path = 'processed/abc123.png';
      const jobId = 'test-job-123';

      const url = generateSignedUrl({ path, jobId });
      const token = url.split('token=')[1];

      const result = validateSignedUrl(token, 'wrong-job-id');
      expect(result.valid).toBe(false);
    });

    it('should reject expired signed URLs', () => {
      const path = 'processed/abc123.png';
      const jobId = 'test-job-123';

      // Generate URL that expires immediately
      const url = generateSignedUrl({ path, jobId, expiresIn: -1 });
      const token = url.split('token=')[1];

      const result = validateSignedUrl(token, jobId);
      expect(result.valid).toBe(false);
      expect(result.error).toContain('expired');
    });

    it('should reject invalid token formats', () => {
      const jobId = 'test-job-123';

      expect(validateSignedUrl('', jobId).valid).toBe(false);
      expect(validateSignedUrl('invalid-token', jobId).valid).toBe(false);
      expect(validateSignedUrl('not-base64!@#$', jobId).valid).toBe(false);
    });

    it('should reject tampered tokens', () => {
      const path = 'processed/abc123.png';
      const jobId = 'test-job-123';

      const url = generateSignedUrl({ path, jobId });
      const token = url.split('token=')[1];

      // Decode, tamper, and re-encode
      const decoded = JSON.parse(Buffer.from(token, 'base64url').toString());
      decoded.path = 'processed/different-file.png';
      const tamperedToken = Buffer.from(JSON.stringify(decoded)).toString('base64url');

      const result = validateSignedUrl(tamperedToken, jobId);
      expect(result.valid).toBe(false);
    });
  });
});
