/**
 * Placeholder pour le modèle custom fine-tuné
 *
 * Ce fichier sera implémenté en Phase 2 pour utiliser
 * votre propre modèle fine-tuné.
 *
 * Options de déploiement:
 * 1. Replicate (custom model) - Recommandé pour commencer
 * 2. Hugging Face Inference Endpoints
 * 3. Self-hosted (avec GPU)
 */

import type { EnhanceOptions } from '@rebloom/shared';
import type { IAIProvider, AIProviderResult } from './provider';
import { aiLogger as logger } from '../../lib/logger';
import { config_ } from '../../lib/config';

export class CustomModelProvider implements IAIProvider {
  name = 'custom' as const;

  constructor() {
    logger.warn('CustomModelProvider is not yet implemented');
  }

  async enhance(_imageUrl: string, _options: EnhanceOptions = {}): Promise<AIProviderResult> {
    const startTime = Date.now();

    // TODO: Implémenter l'appel au modèle custom
    // Exemple pour Replicate custom model:
    //
    // const client = new Replicate({ auth: config_.replicate.apiToken });
    // const output = await client.run(config_.customModel.path, {
    //   input: { image: imageUrl, scale: options.scaleFactor }
    // });

    // Exemple pour Hugging Face Inference:
    //
    // const response = await fetch(`${config_.customModel.endpoint}/predict`, {
    //   method: 'POST',
    //   headers: { 'Authorization': `Bearer ${apiKey}` },
    //   body: JSON.stringify({ image: imageUrl })
    // });

    return {
      success: false,
      error: 'Custom model not implemented yet. See training/ directory for fine-tuning guide.',
      processingTimeMs: Date.now() - startTime,
    };
  }

  async isAvailable(): Promise<boolean> {
    return false;
  }
}

/**
 * Guide pour brancher votre modèle custom:
 *
 * 1. Fine-tuner votre modèle (voir training/FINE_TUNING_GUIDE.md)
 *
 * 2. Déployer sur Replicate:
 *    - Créer un compte Replicate
 *    - Utiliser `cog` pour packager le modèle
 *    - Push vers Replicate: `cog push r8.im/username/model-name`
 *
 * 3. Ou déployer sur Hugging Face:
 *    - Upload le modèle sur HF Hub
 *    - Créer un Inference Endpoint
 *
 * 4. Mettre à jour .env:
 *    CUSTOM_MODEL_ENABLED=true
 *    CUSTOM_MODEL_PATH=username/model-name:version
 *    CUSTOM_MODEL_PROVIDER=replicate
 *
 * 5. Implémenter la méthode enhance() ci-dessus
 */
