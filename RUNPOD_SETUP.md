# RunPod Setup — Diffusers Worker

## 1. Network Volume

- **Bucket** : `sabwi0rezy`
- **Region** : EU-IS-1
- **Taille** : 100GB minimum

## 2. Peupler le volume (une seule fois)

Lancer un **pod temporaire** (CPU ou GPU pas cher) dans la meme region, avec le volume monte.

Dans le terminal du pod :

```bash
wget -O /tmp/dl.py https://raw.githubusercontent.com/steneze/exaether-worker-diffusers-V01/main/download_models.py && python /tmp/dl.py
```

Telecharge ~90GB :
- `lightx2v/Qwen-Image-Edit-Causal` (~57GB) — edit + inpaint
- `Qwen/Qwen-Image` (~30GB) — T2I
- 4 LoRAs Lightning fp32 (4 x 1.7GB)

Verifier a la fin :

```bash
du -sh /runpod-volume/models/*/* /runpod-volume/loras/*
```

Resultat attendu :

```
~57G /runpod-volume/models/lightx2v/Qwen-Image-Edit-Causal
~30G /runpod-volume/models/Qwen/Qwen-Image
1.7G /runpod-volume/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors
1.7G /runpod-volume/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors
1.7G /runpod-volume/loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors
1.7G /runpod-volume/loras/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors
```

**Terminer le pod** une fois le download fini.

## 3. Creer l'endpoint serverless

RunPod Console → Serverless → New Endpoint → **Custom Deployment** :

| Parametre | Valeur |
|-----------|--------|
| Source | GitHub `steneze/exaether-worker-diffusers-V01` (branche `main`) |
| Network Volume | `sabwi0rezy` (EU-IS-1) |
| GPU | A6000 48GB (minimum pour les modeles ~40GB) |
| Min workers | 0 (scale-to-zero, monter a 1 warm quand valide) |
| Max workers | 2 |
| Idle timeout | 5s (defaut) |
| Execution timeout | 300s |

Pas de variable d'environnement requise — les paths par defaut (`/runpod-volume/models`, `/runpod-volume/loras`) correspondent au volume.

## 4. Configurer le backend ExAether

Ajouter dans `.env` du backend :

```
RUNPOD_QWEN_ENDPOINT_ID=<id_de_l_endpoint>
```

Puis lancer le seed :

```bash
cd exaether-api && ./venv/Scripts/python.exe scripts/seed_runpod_diffusers_providers.py
```

Les 3 providers passent en ACTIVE automatiquement.

## 5. Tester

Envoyer une requete via l'API RunPod (ou via le frontend) :

```json
{
  "input": {
    "pipeline": "edit",
    "params": {
      "prompt": "Make the walls white",
      "image1": "<base64>",
      "steps": 8,
      "seed": 42,
      "lora": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors"
    }
  }
}
```

Premier appel : ~2-3min (cold start + model loading). Appels suivants avec meme pipeline : quelques secondes.

## 6. Mettre a jour le worker

Le worker se rebuild automatiquement depuis GitHub a chaque deploy. Pour forcer un rebuild :

RunPod Console → Endpoint → Settings → **Rebuild**.

## Troubleshooting

| Probleme | Solution |
|----------|----------|
| `No such file or directory: /runpod-volume/models/...` | Volume pas monte ou modeles pas telecharges. Verifier avec un pod temporaire. |
| `CUDA out of memory` | Le modele ne rentre pas. Verifier le GPU (A6000 48GB min). |
| Timeout au premier appel | Normal — le lazy loading du premier modele prend 2-3min. Augmenter execution timeout si besoin. |
| LoRA not found | Verifier que le nom de fichier dans la requete correspond exactement au fichier dans `/runpod-volume/loras/`. |
