/**
 * Implémentation du provider Replicate
 * Utilise Real-ESRGAN pour le défloutage d'images
 */

import Replicate from 'replicate';
import { config_ } from '../../lib/config';
import { aiLogger as logger } from '../../lib/logger';
import { ProviderError } from '../../lib/errors';
import { AI_MODELS, DEFAULT_MODEL } from '@rebloom/shared';
import type { EnhanceOptions } from '@rebloom/shared';
import type { IAIProvider, AIProviderResult } from './provider';

export class ReplicateProvider implements IAIProvider {
  name = 'replicate' as const;
  private client: Replicate;

  constructor() {
    this.client = new Replicate({
      auth: config_.replicate.apiToken,
    });
  }

  async enhance(imageUrl: string, options: EnhanceOptions = {}): Promise<AIProviderResult> {
    const startTime = Date.now();
    const { scaleFactor = 4, faceEnhance = false } = options;

    logger.info({ imageUrl, scaleFactor, faceEnhance, model: DEFAULT_MODEL.name }, 'Starting Replicate enhancement');

    try {
      // Construire les inputs selon le modèle
      const isRebloomSharp = DEFAULT_MODEL.id.includes('rebloom-sharp');
      
      const input = isRebloomSharp 
        ? { image: imageUrl }  // ReBloom Sharp n'a qu'un seul input
        : { image: imageUrl, scale: scaleFactor, face_enhance: faceEnhance };

      // Utiliser le modèle configuré
      const output = await this.client.run(DEFAULT_MODEL.id as `${string}/${string}` | `${string}/${string}:${string}`, {
        input,
      });

      const processingTimeMs = Date.now() - startTime;

      // Le résultat peut être une string (URL) ou un array
      const outputUrl = Array.isArray(output) ? output[0] : (output as string);

      if (!outputUrl) {
        throw new Error('No output URL received from Replicate');
      }

      logger.info({ outputUrl, processingTimeMs }, 'Enhancement completed');

      return {
        success: true,
        outputUrl,
        processingTimeMs,
      };
    } catch (error) {
      const processingTimeMs = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      logger.error({ error: errorMessage, processingTimeMs }, 'Enhancement failed');

      return {
        success: false,
        error: errorMessage,
        processingTimeMs,
      };
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      // Vérifier que le modèle ReBloom Sharp est disponible
      const [owner, name] = DEFAULT_MODEL.id.split('/');
      await this.client.models.get(owner, name.split(':')[0]);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Helper pour créer une prédiction avec webhook (async mode)
 */
export async function createReplicatePrediction(
  imageUrl: string,
  options: EnhanceOptions,
  webhookUrl?: string
): Promise<{ predictionId: string }> {
  const client = new Replicate({
    auth: config_.replicate.apiToken,
  });

  const { scaleFactor = 4, faceEnhance = false } = options;
  const isRebloomSharp = DEFAULT_MODEL.id.includes('rebloom-sharp');

  // Input selon le modèle
  const input = isRebloomSharp 
    ? { image: imageUrl }
    : { image: imageUrl, scale: scaleFactor, face_enhance: faceEnhance };

  // Pour ReBloom Sharp (sans version hash), utiliser model au lieu de version
  const predictionConfig: Parameters<typeof client.predictions.create>[0] = isRebloomSharp
    ? {
        model: DEFAULT_MODEL.id as `${string}/${string}`,
        input,
        webhook: webhookUrl,
        webhook_events_filter: ['completed'],
      }
    : {
        version: DEFAULT_MODEL.id.split(':')[1],
        input,
        webhook: webhookUrl,
        webhook_events_filter: ['completed'],
      };

  const prediction = await client.predictions.create(predictionConfig);

  logger.info({ predictionId: prediction.id, model: DEFAULT_MODEL.name }, 'Replicate prediction created');

  return { predictionId: prediction.id };
}

/**
 * Helper pour récupérer le statut d'une prédiction
 */
export async function getReplicatePrediction(predictionId: string) {
  const client = new Replicate({
    auth: config_.replicate.apiToken,
  });

  return client.predictions.get(predictionId);
}
