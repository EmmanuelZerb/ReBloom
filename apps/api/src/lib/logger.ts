/**
 * Logger configuration avec Pino
 * Logs structurés pour production
 */

import pino from 'pino';
import { env } from './config';

export const logger = pino({
  level: env.NODE_ENV === 'development' ? 'debug' : 'info',
  transport:
    env.NODE_ENV === 'development'
      ? {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: 'HH:MM:ss',
            ignore: 'pid,hostname',
          },
        }
      : undefined,
  base: {
    service: 'rebloom-api',
    env: env.NODE_ENV,
  },
});

// Child loggers pour différents modules
export const createLogger = (module: string) => logger.child({ module });

export const apiLogger = createLogger('api');
export const jobLogger = createLogger('job');
export const storageLogger = createLogger('storage');
export const aiLogger = createLogger('ai');
