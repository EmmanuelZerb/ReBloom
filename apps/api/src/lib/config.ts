/**
 * Configuration centralisée de l'API
 * Charge les variables d'environnement avec validation
 */

import { config } from 'dotenv';
import { z } from 'zod';
import path from 'path';

// Load .env file from root
config({ path: path.resolve(process.cwd(), '../../.env') });

const envSchema = z.object({
  // Server
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  API_PORT: z.coerce.number().default(3001),
  API_HOST: z.string().default('localhost'),

  // Redis
  REDIS_URL: z.string().default('redis://localhost:6379'),

  // Replicate
  REPLICATE_API_TOKEN: z.string().min(1, 'REPLICATE_API_TOKEN is required'),

  // Storage
  STORAGE_PROVIDER: z.enum(['local', 's3']).default('local'),
  STORAGE_LOCAL_PATH: z.string().default('./uploads'),

  // S3 (optional)
  S3_BUCKET: z.string().optional(),
  S3_REGION: z.string().optional(),
  S3_ACCESS_KEY_ID: z.string().optional(),
  S3_SECRET_ACCESS_KEY: z.string().optional(),
  S3_ENDPOINT: z.string().optional(),

  // Job config
  JOB_TIMEOUT_MS: z.coerce.number().default(300000),
  JOB_MAX_RETRIES: z.coerce.number().default(3),

  // Upload limits
  MAX_FILE_SIZE_MB: z.coerce.number().default(20),

  // Webhook
  WEBHOOK_URL: z.string().optional(),
  WEBHOOK_SECRET: z.string().optional(),

  // Custom model
  CUSTOM_MODEL_ENABLED: z.coerce.boolean().default(false),
  CUSTOM_MODEL_PATH: z.string().optional(),
  CUSTOM_MODEL_PROVIDER: z.enum(['replicate', 'huggingface', 'local']).default('replicate'),
});

function loadConfig() {
  const parsed = envSchema.safeParse(process.env);

  if (!parsed.success) {
    console.error('❌ Invalid environment variables:');
    console.error(parsed.error.flatten().fieldErrors);
    throw new Error('Invalid environment configuration');
  }

  return parsed.data;
}

export const env = loadConfig();

export const config_ = {
  isDev: env.NODE_ENV === 'development',
  isProd: env.NODE_ENV === 'production',

  server: {
    port: env.API_PORT,
    host: env.API_HOST,
  },

  redis: {
    url: env.REDIS_URL,
  },

  replicate: {
    apiToken: env.REPLICATE_API_TOKEN,
  },

  storage: {
    provider: env.STORAGE_PROVIDER,
    localPath: env.STORAGE_LOCAL_PATH,
    s3: {
      bucket: env.S3_BUCKET,
      region: env.S3_REGION,
      accessKeyId: env.S3_ACCESS_KEY_ID,
      secretAccessKey: env.S3_SECRET_ACCESS_KEY,
      endpoint: env.S3_ENDPOINT,
    },
  },

  jobs: {
    timeoutMs: env.JOB_TIMEOUT_MS,
    maxRetries: env.JOB_MAX_RETRIES,
  },

  upload: {
    maxSizeBytes: env.MAX_FILE_SIZE_MB * 1024 * 1024,
  },

  webhook: {
    url: env.WEBHOOK_URL,
    secret: env.WEBHOOK_SECRET,
  },

  customModel: {
    enabled: env.CUSTOM_MODEL_ENABLED,
    path: env.CUSTOM_MODEL_PATH,
    provider: env.CUSTOM_MODEL_PROVIDER,
  },
} as const;

export type Config = typeof config_;
