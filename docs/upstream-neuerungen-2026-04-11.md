# llama.cpp Neuerungen - Übersicht für Sascha

**Datum**: 11. April 2026  
**Repository**: https://github.com/ggml-org/llama.cpp  
**Ziel**: Identifiziere beachtenswerte Neuerungen für den eigenen Fork

---

## 🚀 Wichtige Performance-Optimierungen (PRs)

### 🔴 KRITISCH für CUDA Multi-GPU

#### 1. CUDA: NCCL comms lazily initialize (PR #21746)
- **Problem**: NCCL comms nehmen hunderte MB VRAM pro Gerät
- **Lösung**: NCCL comms werden lazy und pro device-vector initialisiert
- **Impact**: Reduziert VRAM usage signifikant bei Multi-GPU setups
- **Author**: Johannes Gäßler
- **Status**: OPEN

#### 2. CUDA: Limit DeviceSegmentedSort (PR #21718)
- **Problem**: DeviceSegmentedSort ist nicht capturable in CUDA graphs
- **Lösung**: Beschränke auf immediate mode, nutze RadixSort für graphs
- **Impact**: Bessere CUDA graph compatibility für argsort operations
- **Performance**: DeviceSegmentedRadixSort ~40% schneller in graph mode
- **Author**: Oliver Simons

### 🟡 Wichtige CUDA Optimierungen

#### 3. CUDA graph equality fix (PR #21736)
- **Problem**: node->src ne/nb nicht gespeichert, graph equality check fehlerhaft
- **Lösung**: Speichert zusätzlich ne/nb für graph comparison
- **Impact**: Verbesserte CUDA graph stability und correctness
- **Performance**: Kein messbarer Unterschied (+/- 1%)
- **Author**: Aman Gupta

#### 4. CUDA: VRAM to LDS loading pipeline (PR #21698)
- **Problem**: Inefficient VRAM to LDS loading in load_tiles_q8_0
- **Lösung**: Besseres Pipeline für tiled loading
- **Impact**: Verbesserte Performance für Q8_0 quantization
- **Author**: Deneb1312

### 🟢 Wichtige Server Optimierungen

#### 5. Server: SWA prompt caching fix (PR #21749)
- **Problem**: SWA models force full prompt re-processing auf jedem Request
- **Lösung**: 
  - pos_min_thold = 0 when swa_full enabled
  - Skip checkpoint restoration when swa_full enabled
- **Impact**: Bessere Performance für SWA models mit prompt caching
- **Author**: shipped-it
- **Closes**: #21468

---

## 🐛 Wichtige Bug Fixes (Issues)

### 🔴 CRITICAL

#### 6. Gemma 4 VRAM memory leak (Issue #21742)
- **Problem**: Gemma 4 finetunes leak VRAM
- **Status**: OPEN
- **Priority**: HIGH (production critical)

#### 7. Server crash bei großen Bildern (Issue #21750)
- **Problem**: Server crash mit >4B Qwen3.5 und Qwen3-VL models
- **Status**: OPEN
- **Priority**: HIGH (production critical)

#### 8. NCCL stuck bei Teilmenge von GPUs (Issue #21719)
- **Problem**: NCCL communications stuck bei Submenge von CUDA GPUs
- **Status**: OPEN
- **Priority**: HIGH (Multi-GPU setups affected)
- **Related PR**: #21746 (NCCL lazy init soll dieses fixen)

#### 9. llama-bench Fehler (Issue #21748)
- **Problem**: llama-bench fails mit -c oder --ctx-size
- **Status**: OPEN
- **Priority**: MEDIUM (breaking tool functionality)

#### 10. Gemma4 gibberish mit -nkvo (Issue #21726)
- **Problem**: Gemma4 models produce gibberish mit -nkvo Flag
- **Status**: OPEN
- **Priority**: MEDIUM (incorrect outputs)

### 🟡 MODERATE

#### 11. Server gibberish bei zweiter Turn (Issue #21734)
- **Problem**: Server-intel produces gibberish after first turn
- **Status**: OPEN
- **Priority**: MEDIUM (quality issue)

#### 12. Docker CUDA 13 nur 60% GPU (Issue #21740)
- **Problem**: Docker CUDA 13 image nur 60% GPU nutzt
- **Status**: OPEN
- **Priority**: LOW (environment-specific)

#### 13. SYCL crash (Issue #21747)
- **Problem**: Tool crashes beim Build mit SYCL
- **Status**: OPEN
- **Priority**: LOW (SYCL users)

---

## 🎯 Feature Requests (Potential Value)

#### 14. Metal/GGML: KV-Cache Pruning Porting (Issue #21743)
- **Feature**: Port Advanced KV-Cache Pruning (H2O, Rocket, Evol-KV) zu Metal
- **Status**: Feature Idea
- **Potential**: H2O pruning ist sehr effektiv (20-30% speedup)
- **Priority**: HIGH (könnte zu HISA parallelisiert werden)

#### 15. TileLang Acceleration Support (Feature #21712)
- **Feature**: TileLang Acceleration Support für llama.cpp
- **Status**: Feature Request
- **Priority**: LOW (experimentell, noch nicht production-ready)

#### 16. XDNA Backend (Feature #21725)
- **Feature**: XDNA backend für mobile GPUs
- **Status**: Feature Request
- **Priority**: LOW (Niche use case)

---

## 🌐 Backend-Spezifische Neuerungen

### Vulkan
- **Q4_K/Q5_K scale loads optimized** (PR #21751)
  - Bessere SPIR-V compiler compatibility (mesa/Intel)
  - +4% bis +10% performance auf Arc GPUs
  - *Hinweis: Vulkan ist nicht primärer Fokus, aber interessant*

### WebGPU
- **Windows D3D12 fallback** (PR #21744)
- **Matrix-vector multiplication updated** (PR #21738)

### SYCL
- **Q8_0 reorder optimization** (Commit 0988acc)
- **Bug**: Gibberish nach diesem Commit (Issue #21715)
- **Priority**: LOW (SYCL niche)

---

## 📊 Empfehlungen für den Fork

### 🥇 TOP PRIORITY (Sofort einbauen)

1. **CUDA: NCCL lazy init** (PR #21746)
   - Fixes VRAM leak und AllReduce stuck
   - Kann Probleme mit Multi-GPU setups verhindern

2. **Gemma 4 VRAM leak** (Issue #21742)
   - Critical bug für Gemma 4 users
   - HINWEIS: Unsere HISA changes könnten damit zusammenhängen

3. **Server SWA caching fix** (PR #21749)
   - Verbessert Performance für SWA models
   - Kombinierbar mit HISA

### 🥈 HIGH PRIORITY (Nächste Sprints)

4. **CUDA argsort limit** (PR #21718)
   - Bessere CUDA graph compatibility
   - Performance improvement für argsort-heavy workloads

5. **CUDA graph equality fix** (PR #21736)
   - Verbesserte stability
   - Niedriges Risiko

6. **llama-bench fix** (Issue #21748)
   - Broken tool functionality
   - Einfach zu reproduzieren und zu fixen

### 🥉 MEDIUM PRIORITY (Optional)

7. **Vulkan Q4_K/Q5_K optimization** (PR #21751)
   - +4% bis +10% performance auf Arc GPUs
   - Nur interessant wenn Vulkan supported wird

8. **KV-Cache Pruning Porting** (Issue #21743)
   - H2O/Rocket/Evol-KV Port zu Metal
   - Kann zu HISA parallelisiert werden

---

## 🔧 Merge-Hinweise

### NCCL lazy init (#21746)
- Keine breaking changes
- Backward compatible
- Niedriges Risiko

### CUDA graphs fixes (#21736, #21718)
- Nur CUDA backend
- Low risk
- Test mit CUDA graphs erforderlich

### SWA caching (#21749)
- Server feature
- Low risk
- Test mit SWA models erforderlich

---

## 📚 Weitere Ressourcen

- **Upstream Repo**: https://github.com/ggml-org/llama.cpp
- **Issue Tracker**: https://github.com/ggml-org/llama.cpp/issues
- **Pull Requests**: https://github.com/ggml-org/llama.cpp/pulls
- **Discussions**: https://github.com/ggml-org/llama.cpp/discussions

---

**Zusammenfassung**:
- **10 offene PRs** mit hoher Impact
- **13 offene Issues** mit varying severity
- **3 Feature Requests** mit interessanten Ideen
- **3 Backend-Spezifische Neuerungen** (Vulkan, WebGPU, SYCL)

Empfohlene Strategie:
1. Priorisiere CUDA fixes (NCCL, graphs, argsort)
2. Fixe Gemma 4 VRAM leak (critical production bug)
3. Teste HISA integration mit neuen commits
4. Berücksichtige Metal KV-pruning parallel zu HISA
