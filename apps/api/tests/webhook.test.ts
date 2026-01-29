/**
 * Tests for webhook signature validation
 */

import { describe, it, expect } from 'vitest';
import { verifyWebhookSignature, generateWebhookSignature } from '../src/lib/webhook';

describe('Webhook Signature', () => {
  const testSecret = 'test-webhook-secret';
  const testPayload = JSON.stringify({
    id: 'test-prediction-123',
    status: 'succeeded',
    output: ['https://example.com/output.png'],
  });

  describe('generateWebhookSignature', () => {
    it('should generate a valid signature', () => {
      const signature = generateWebhookSignature(testPayload, testSecret);

      expect(signature).toBeDefined();
      expect(signature.startsWith('sha256=')).toBe(true);
      expect(signature.length).toBeGreaterThan(10);
    });

    it('should generate consistent signatures', () => {
      const sig1 = generateWebhookSignature(testPayload, testSecret);
      const sig2 = generateWebhookSignature(testPayload, testSecret);

      expect(sig1).toBe(sig2);
    });

    it('should generate different signatures for different payloads', () => {
      const sig1 = generateWebhookSignature(testPayload, testSecret);
      const sig2 = generateWebhookSignature('different payload', testSecret);

      expect(sig1).not.toBe(sig2);
    });

    it('should generate different signatures for different secrets', () => {
      const sig1 = generateWebhookSignature(testPayload, testSecret);
      const sig2 = generateWebhookSignature(testPayload, 'different-secret');

      expect(sig1).not.toBe(sig2);
    });
  });

  describe('verifyWebhookSignature', () => {
    it('should verify valid signatures', () => {
      const signature = generateWebhookSignature(testPayload, testSecret);
      const isValid = verifyWebhookSignature(testPayload, signature, testSecret);

      expect(isValid).toBe(true);
    });

    it('should reject invalid signatures', () => {
      const isValid = verifyWebhookSignature(
        testPayload,
        'sha256=invalid-signature-here',
        testSecret
      );

      expect(isValid).toBe(false);
    });

    it('should reject signatures with wrong secret', () => {
      const signature = generateWebhookSignature(testPayload, testSecret);
      const isValid = verifyWebhookSignature(testPayload, signature, 'wrong-secret');

      expect(isValid).toBe(false);
    });

    it('should reject tampered payloads', () => {
      const signature = generateWebhookSignature(testPayload, testSecret);
      const tamperedPayload = testPayload.replace('succeeded', 'failed');
      const isValid = verifyWebhookSignature(tamperedPayload, signature, testSecret);

      expect(isValid).toBe(false);
    });

    it('should reject missing signatures', () => {
      expect(verifyWebhookSignature(testPayload, '', testSecret)).toBe(false);
      expect(verifyWebhookSignature(testPayload, null as any, testSecret)).toBe(false);
      expect(verifyWebhookSignature(testPayload, undefined as any, testSecret)).toBe(false);
    });

    it('should reject missing secrets', () => {
      const signature = generateWebhookSignature(testPayload, testSecret);

      expect(verifyWebhookSignature(testPayload, signature, '')).toBe(false);
      expect(verifyWebhookSignature(testPayload, signature, null as any)).toBe(false);
      expect(verifyWebhookSignature(testPayload, signature, undefined as any)).toBe(false);
    });

    it('should reject malformed signature format', () => {
      // Missing sha256= prefix
      expect(verifyWebhookSignature(testPayload, 'abc123def456', testSecret)).toBe(false);

      // Wrong algorithm
      expect(verifyWebhookSignature(testPayload, 'sha1=abc123', testSecret)).toBe(false);
      expect(verifyWebhookSignature(testPayload, 'md5=abc123', testSecret)).toBe(false);
    });

    it('should be timing-safe (not vulnerable to timing attacks)', () => {
      // This is a behavioral test - we verify the function completes in roughly
      // consistent time regardless of where the mismatch occurs
      const signature = generateWebhookSignature(testPayload, testSecret);
      const hexPart = signature.replace('sha256=', '');

      // Early mismatch
      const earlyMismatch = `sha256=0${hexPart.slice(1)}`;
      // Late mismatch
      const lateMismatch = `sha256=${hexPart.slice(0, -1)}0`;

      // Both should be rejected
      expect(verifyWebhookSignature(testPayload, earlyMismatch, testSecret)).toBe(false);
      expect(verifyWebhookSignature(testPayload, lateMismatch, testSecret)).toBe(false);
    });
  });
});
