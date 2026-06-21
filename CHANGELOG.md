# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Bug Fixes

- **docs**: Work around Sphinx 9.1 _MockObject annotations breaking get_type_hints ([b4817f7](https://github.com/experimaestro/xpm-torch/commit/b4817f768340563bd10e2d71dc1f90ae8a64857a))- Make _from_pretrained signature optional for hf-hub compatibility ([55e5fcf](https://github.com/experimaestro/xpm-torch/commit/55e5fcfd8b80ccadaf3ea840b10ced86a9451743))
- Ruff ([dab1d27](https://github.com/experimaestro/xpm-torch/commit/dab1d27583dc31e9e8730417164d12b01c4e82f8))
- Update experimaestro for HF support ([5c313ed](https://github.com/experimaestro/xpm-torch/commit/5c313ed9a40a5e97f2aa50d6ce36567ecc1ad550))
- Use fabric gradient clipping method ([f3ff6e5](https://github.com/experimaestro/xpm-torch/commit/f3ff6e5818fab6861060fbd08317129f9415d7fa))
- Logging fabric params ([f090311](https://github.com/experimaestro/xpm-torch/commit/f090311c200dd7292c416307a1270222658b6375))
- No warning when using single devices ([09d77f8](https://github.com/experimaestro/xpm-torch/commit/09d77f83404b409b3dd42f2ce36a3e97f19268a0))
- Robust dir serialization for SimpleModuleLoader ([57e038e](https://github.com/experimaestro/xpm-torch/commit/57e038ef536fef2871ad47c43c5d9295e0f605d9))
- To_device generalization ([3180f10](https://github.com/experimaestro/xpm-torch/commit/3180f1094c68d2a6dfde0c11aa79dcb4189c7cd4))
- Proper kwarg for clipping ([f594f0a](https://github.com/experimaestro/xpm-torch/commit/f594f0af28a44db102af10347b2f5ad587417fb8))
- Remove unused Optional import to unblock CI ([6df8f67](https://github.com/experimaestro/xpm-torch/commit/6df8f67b372d5992705f4fb12d25bfd2bfa58c2b))

### Features
- Default serialize safetensors to folder root ([8078fbf](https://github.com/experimaestro/xpm-torch/commit/8078fbfca8b83ee17156a55e2b5c15e252a210c5))
- User-frendly interface for TorchHFHub models ([ecf7146](https://github.com/experimaestro/xpm-torch/commit/ecf7146a646b210cd05694d9fd712a2d3f978bd6))
- Get_foward_methods for Module ([9030d87](https://github.com/experimaestro/xpm-torch/commit/9030d87431bcf5f9f53914e0a527c6c7473f2285))
- Store self.fabric during setup for later dataloader setup ([ea0b36d](https://github.com/experimaestro/xpm-torch/commit/ea0b36d4fbe7e9455f119362c2528280edf412f4))
- Improved to_device helper to be recursive ([ddfb7f1](https://github.com/experimaestro/xpm-torch/commit/ddfb7f19825fcd34a15b214ed4a1b3c201ff7089))
- Logging error if torch.distributed not initialized ([5f0ac90](https://github.com/experimaestro/xpm-torch/commit/5f0ac90e48b8db7e2e3dd5967aabbb381c7b0bef))
- Add PredictiveBatcher with pre-DDP profile phase ([07adc36](https://github.com/experimaestro/xpm-torch/commit/07adc36c8a3829aac54432c547754e1ee60bf4d4))

### Miscellaneous Tasks
- Add uv.lock for CI caching ([c0a5a93](https://github.com/experimaestro/xpm-torch/commit/c0a5a931ff6d718c4cd4bf8dd60bcdad5aa45166))
- Added version to GradientClippingHook ([b175ad4](https://github.com/experimaestro/xpm-torch/commit/b175ad4303b9eece6a326f99d3d8516073f5047b))
## [0.1.0] - 2026-03-30

### Bug Fixes
- Fixed MRO of xpmTorchModule ([758ad15](https://github.com/experimaestro/xpm-torch/commit/758ad1583281f142c40e7484e8cdb8db53f0d63d))
- Fixed last imports ([b21870c](https://github.com/experimaestro/xpm-torch/commit/b21870c5d6bea7245aeeb1254f649f342e96b2f7))
- Fixed install, fields and tensorboard ([7b93987](https://github.com/experimaestro/xpm-torch/commit/7b939873a6ed08a866506e10677455807c629635))
- Fixed foreach ([9f84962](https://github.com/experimaestro/xpm-torch/commit/9f84962a971c7bba9579f46f20f3e8062802f2d4))
- Fixed symlink to tensorboard service ([38d7c46](https://github.com/experimaestro/xpm-torch/commit/38d7c46b9a1d81cce45679e448d020831629e608))
- Fixed dummy_param loading ([150c425](https://github.com/experimaestro/xpm-torch/commit/150c425f9518dac92e6736449fbef92975dbc450))
- Fixed typo and merge for hf ([2885d88](https://github.com/experimaestro/xpm-torch/commit/2885d888bc05e4358d44c570acc25cf8f399997c))
- Correct hooks into hook ([995bde8](https://github.com/experimaestro/xpm-torch/commit/995bde855524d3a06552c5c46396abbd884542a2))
- Bump to experimaestro 2.0.7 for services ([cd320a4](https://github.com/experimaestro/xpm-torch/commit/cd320a465a86fafcfe711ef6da5610daa8aa841a))
- Fix import ([9c5498b](https://github.com/experimaestro/xpm-torch/commit/9c5498ba97fdac12a435f0965680017f63ef2971))
- Fixed dataloader state dict ([9953e8b](https://github.com/experimaestro/xpm-torch/commit/9953e8b3085ee612c9ffa1ad650f4d8a292d8f83))
- Fixed other Trainers ([007d068](https://github.com/experimaestro/xpm-torch/commit/007d0684bb5e22076395097991bc0eeaed62a5b9))
- Fix dataloader loading ([e8513a4](https://github.com/experimaestro/xpm-torch/commit/e8513a4def73a5e3d83fba70475e5391a0797e31))
- Formatting ([4886c2c](https://github.com/experimaestro/xpm-torch/commit/4886c2c34380b34d9e7ccf676a0347f403999b9f))
- Convert DataPath to Path in ModuleLoader.execute ([52dfd66](https://github.com/experimaestro/xpm-torch/commit/52dfd66ea4f28bca10e168e4d754ad8c19a2b005))
- Clean up unused imports, add ruff to CI deps ([53de88b](https://github.com/experimaestro/xpm-torch/commit/53de88bbfb0e87329bf5113bd61214962daee5ce))
- Resolve all ruff lint errors ([fa62018](https://github.com/experimaestro/xpm-torch/commit/fa620182bd9f562c3c5038eeb4ca78febc409db0))
- Batcher was not used properly ([4a51e80](https://github.com/experimaestro/xpm-torch/commit/4a51e80d139227510e8b70910719579b3a0ca441))
- Changed Batchers to be Meta ([8c62f04](https://github.com/experimaestro/xpm-torch/commit/8c62f0429515f550ae51d23ff88063d68b51b80f))

### CI
- Add GitHub Actions, tests, and pre-commit config ([a949b9f](https://github.com/experimaestro/xpm-torch/commit/a949b9fabd17803e59c93f7ad400a336db149e6f))

### Documentation
- Add Sphinx documentation with full API coverage ([9bebe77](https://github.com/experimaestro/xpm-torch/commit/9bebe77f5f0b92560c69326bf9273d1def64c910))
- Update for new AutoModel API and ReadmeSection ([f80a8ee](https://github.com/experimaestro/xpm-torch/commit/f80a8ee416d7aa5defbb205bf9ae61417e5e9555))
- Update for settings pattern, add ExportAction ([4b764ba](https://github.com/experimaestro/xpm-torch/commit/4b764ba23ad7cb7605722bcc65c10fd51c584d33))
- Fix a typo ([a34fea1](https://github.com/experimaestro/xpm-torch/commit/a34fea105d7247ecbd8899427bcf9298917de77e))

### Features
- Convert TensorBoard to ProcessWebService and simplify symlinks ([8490800](https://github.com/experimaestro/xpm-torch/commit/849080013f1802cf06fceb0e5d705f08c4d5eed4))
- Add initialized decorator, remove explicit model.initialize from Learner ([8ff3d95](https://github.com/experimaestro/xpm-torch/commit/8ff3d9581f54ff3d7d64277aa732b3f60d9856f6))
- Use standard TensorBoard port (6006+) instead of random port ([2431ceb](https://github.com/experimaestro/xpm-torch/commit/2431ceb9a2f0ca5cddb779d1a81793726440ae83))
- Better logging for fabric setup ([31f82ef](https://github.com/experimaestro/xpm-torch/commit/31f82efd2fae1f3b8e954d7751fd6fb80086d367))
- Make Sampler generic with SampleT type parameter ([6159ac2](https://github.com/experimaestro/xpm-torch/commit/6159ac28be8e20abc0a85abb14dd046c4d239f93))
- Migrated ScorerOutputType ([a1821c0](https://github.com/experimaestro/xpm-torch/commit/a1821c02cbc2db100338f929ae8980de14e437f9))
- Safetensors checkpointing for model save/load ([018ce25](https://github.com/experimaestro/xpm-torch/commit/018ce25b9e35be2ac015354286e03b9c0f1f14cd))
- Add loader_config pattern, refactor ModuleLoader wrappers ([892d8ba](https://github.com/experimaestro/xpm-torch/commit/892d8bac252a5a963d384da13b7f6dd37181c474))
- Add TrainingResults config and upload-hfhub CLI ([1bfbb88](https://github.com/experimaestro/xpm-torch/commit/1bfbb88d7030c85342caad5f4e88849f5c3fd598))
- Add write_hub_extras base method on Module ([ddcc5aa](https://github.com/experimaestro/xpm-torch/commit/ddcc5aa3e874d4b5796ec4a9e8c58abf3564dd7f))
- Add hub_readme_extra() method on Module ([3688a9b](https://github.com/experimaestro/xpm-torch/commit/3688a9b1bfa1c987074eea100e3453b74fe37fcc))
- Template-based README with ordered sections, model property on ModuleLoader ([9b97ed8](https://github.com/experimaestro/xpm-torch/commit/9b97ed8d186ace0b308365c00e062a860dff5f15))
- Add ExportAction, migrate Learner to __submit__ ([baab5e0](https://github.com/experimaestro/xpm-torch/commit/baab5e02f3fb1e472cf635291908ad3b12ae000c))
- Add Module.export_action for model-controlled action creation ([a4de398](https://github.com/experimaestro/xpm-torch/commit/a4de3988bead8a7d993267a92480a9692fb7bdd0))
- Added experiment configuration ([e813397](https://github.com/experimaestro/xpm-torch/commit/e813397b3d40c34d7bfb120deea31098de7a22c7))

### Miscellaneous Tasks
- Clean up unused imports ([31f9021](https://github.com/experimaestro/xpm-torch/commit/31f902175287e3e239d871d787ceb1f6ac738087))
- Update pyproject.toml with CLI entry, dependency groups, and docs deps ([419f723](https://github.com/experimaestro/xpm-torch/commit/419f723a8e8dc8e33c2d1174de53eb043bd2593e))
- Add dev dependency group with ruff ([d7b846c](https://github.com/experimaestro/xpm-torch/commit/d7b846c672dd84e91a8c80e42a99d933019c4f80))
- Bump experimaestro requirement to >=2.3 ([df8f127](https://github.com/experimaestro/xpm-torch/commit/df8f1273423f2360de5c516ba685993b530a9a21))
- Added ignored files ([b5f2732](https://github.com/experimaestro/xpm-torch/commit/b5f2732b5fa7e039c2fb7e78a4b7edf8e0e8a19c))
- Added test requirements ([4bc935d](https://github.com/experimaestro/xpm-torch/commit/4bc935dbca0d9f4df132b52a3d3d96a96ed803bb))

### Refactor
- Remove ModuleInitMode, make initialization explicit ([76729b0](https://github.com/experimaestro/xpm-torch/commit/76729b03ebb8ee4bb77b26637f82ba9bb659a29a))
- Remove get_collate_fn and HydratingCollate ([8311cd4](https://github.com/experimaestro/xpm-torch/commit/8311cd4169eadaaa9558fd5ac8c00ff003284625))
- Remove Record import from trainers ([3625d1d](https://github.com/experimaestro/xpm-torch/commit/3625d1d0dbbeece3fc35f5fe3956b088cdf5f55c))
- Refactor and better doctrings ([316fcaf](https://github.com/experimaestro/xpm-torch/commit/316fcaff910b187ae0c0251319a04463c1c4a3c7))
- Rename get_Fabric to get_fabric, add FabricConfigurationBase ([66147ff](https://github.com/experimaestro/xpm-torch/commit/66147ff2de7c68e5ce4f3cb309aa367dfccbd246))
- Migrate bare default values to field(default=..., ignore_default=True) ([f7ac18e](https://github.com/experimaestro/xpm-torch/commit/f7ac18ebbce8bda4f0b304b9d852411ccce4a63d))
- Consolidate HF code, remove xpmTorchHubModule ([40256be](https://github.com/experimaestro/xpm-torch/commit/40256beb5b5cf40a82b1aaad5d1a27f23a4b644c))
- Move write_hub_extras/hub_readme_extra from Module to ModuleLoader ([0837431](https://github.com/experimaestro/xpm-torch/commit/08374310fce20899f5765f55932a279477c0933b))
- ModuleLoader hierarchy, TorchHFHub, template README ([97e3656](https://github.com/experimaestro/xpm-torch/commit/97e36561a80bc74b21252ba9a4b676c91f234e0e))
- ModuleLoader carries settings, remove wrapper classes ([a189ced](https://github.com/experimaestro/xpm-torch/commit/a189cedfec584e3503122f8dbbee35a8421eb515))
- Move xpmir-dependent trainers to xpmir, add huggingface-hub dep ([00d1ff2](https://github.com/experimaestro/xpm-torch/commit/00d1ff22c68a9bdf6afbd7e89516e67dfd9158a6))
- Remove upload-hfhub CLI command, document action-based export ([6e40b04](https://github.com/experimaestro/xpm-torch/commit/6e40b04f28a238b8e98977f75d5e977f675c9aa5))

### Testing
- Add astral/uv cache ([41d6f32](https://github.com/experimaestro/xpm-torch/commit/41d6f3233b2db38dd5c4718688ccf5add8ad0a07))

### Wip
- Moving from XPM-IR ([5c60c83](https://github.com/experimaestro/xpm-torch/commit/5c60c837c9f33d80f4edc08f70df997fa9d06c80))

