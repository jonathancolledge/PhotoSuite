# models

Downloaded models go here. Nothing in this folder is part of the repository; see [THIRD_PARTY.md](../THIRD_PARTY.md)
for each model's source and licence.

| Folder | App | Filled by |
|---|---|---|
| `PhotoLibrary/` | Photo Library: `buffalo_l` (faces), `siglip2`, `florence2` | Settings & AI → Download missing models, or `PhotoLibrary/desktop/photolibrary_setup_models.py` |
| `RAWDemosaic/` | RAW Demosaic | `RAWDemosaic/desktop/setup_models.py` |
| `panoramamaker/` | PANOSTITCH: `raco_aliked_lightglue_pipeline_k*.onnx` | `panoramamaker/desktop/panostitch_setup_models.py` |

The installers' `-Models` (Windows) and `--models` (Linux) options run all three.

The browser versions use the same folders: `download_web_models.py` (in the top folder) adds the ONNX models
`rawdemosaic.html` needs to `RAWDemosaic/`, and the ALIKED models to `panoramamaker/`.
