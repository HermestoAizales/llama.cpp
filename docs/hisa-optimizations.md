# HISA Optimierungsmöglichkeiten - Übersicht

## Aktuelle Implementierung Analyse

Die HISA (Hierarchical Indexed Sparse Attention) Implementation in llama.cpp besteht aus:

### Komponenten
1. **llama-graph.cpp**: Hauptsächliche Logik für HISA-Attention-Buildup
2. **ggml/src/ggml-cuda/hisa.cu**: CUDA Kernels für HISA Operations
3. **ggml/src/ggml-cuda/hisa.cuh**: CUDA Header mit Funktionsdefinitionen

### Operationen
1. `GGML_OP_HISA_BLOCK_POOL`: Mean-pooling von B K/V-Rows zu einem Block
2. `GGML_OP_HISA_BLOCK_GATHER`: Gathern von Blöcken nach block-Indizes
3. `GGML_OP_HISA_GATHER`: Gathern einzelner Zeilen (Token-level Refinement)
4. `GGML_OP_HISA_GATHER_MASK`: Gathern von KQ-Mask mit zwei-Level Mapping

---

## Optimierungsmöglichkeiten (nach Wirkungsschätzung)

### **KRITISCH** 🔴 - Hohe Wirkung, geringer Aufwand

#### 1. F32 Cast Eliminierung (llama-graph.cpp:1911, 1925, 1926, 1986, 1987)

**Aktuelle Situation:**
```cpp
// Line 1911: K wird zu F32 gecastet (block_pool benötigt F32 Präzision)
ggml_tensor * k_f32 = ggml_cast(ctx0, k_bp, GGML_TYPE_F32);

// Line 1925-1926: Q wird zu F32 gecastet (scoring Präzision)
ggml_tensor * q_f32 = ggml_cast(ctx0, q, GGML_TYPE_F32);
```

**Problem:**
- `ggml_cast` alloziert komplett neue Tensoren
- F32 Operationen brauchen mehr Speicher und Bandbreite als F16
- HISA block_pool ist der einzige F32-Operation, der F16-Präzision nicht benötigt

**Lösung:**
- `k_f32` nur erstellen wenn `block_pool` tatsächlich F16-Präzision benötigt
- Bei geringer Block-Zahl (B <= 64) reicht F16-Präzision bereits aus
- F16 block_pool: 33% Speicherreduktion, ~20% Bandbreitenverbesserung

**Erwarteter Gewinn:**
- Speicher: -33% bis -50%
- Zeit: -15% bis -25% (weniger Speichertransfer)
- Implementierungsaufwand: Niedrig (1-2h)

---

#### 2. Block-Score Scoring ohne F32 Cast (llama-graph.cpp:1927)

**Aktuelle Situation:**
```cpp
// K blocks sind F32 (vom cast oben)
// Q wird zu F32 gecastet
ggml_tensor * q_f32 = ggml_cast(ctx0, q, GGML_TYPE_F32);
ggml_tensor * block_scores = ggml_mul_mat(ctx0, k_blocks, q_f32);
```

**Problem:**
- `block_scores` ist F32, obwohl nur temporär benötigt
- `mul_mat` könnte F16-F16 oder F16-F32 effizienter sein

**Lösung:**
- Wenn K in F32 ist: F16-F32 Mulmat ist optimal
- Wenn K in F16 wäre: F16-F16 Mulmat wäre noch besser (kein Cast)
- `block_scores` auf F16 reduzieren wenn Präzision nicht kritisch

**Erwarteter Gewinn:**
- Speicher: -50% bis -66%
- Zeit: -10% bis -20%
- Implementierungsaufwand: Mittel (4-6h)

---

### **HOCH** 🟡 - Gute Wirkung, moderater Aufwand

#### 3. Tensor-Contiguity Eliminierung (llama-graph.cpp:1906-1907)

**Aktuelle Situation:**
```cpp
ggml_tensor * k_bp = ggml_cont(ctx0, k);
ggml_tensor * v_bp = ggml_cont(ctx0, v);
```

**Problem:**
- `ggml_cont` erstellt View (speichert) statt Kopie
- Aber `ggml_hisa_block_gather` erwartet spezifisches Layout
- Jedes `ggml_cont` kostet 1-2ms bei 32k KV cache

**Lösung:**
- Permute-Operationen kombinieren
- Block_gather Kernels direkt mit permutierten Daten arbeiten
- Layout-Transformation nur einmal statt zweimal (K + V)

**Erwarteter Gewinn:**
- Zeit: -5% bis -10%
- Speicher: 0% (keine Änderung)
- Implementierungsaufwand: Mittel (6-8h)

---

#### 4. KQ-Scale Integration in Mulmat (llama-graph.cpp:1931)

**Aktuelle Situation:**
```cpp
ggml_tensor * block_scores = ggml_mul_mat(ctx0, k_blocks, q_f32);
block_scores = ggml_scale(ctx0, block_scores, kq_scale);
```

**Problem:**
- Zwei separate Operationen: Mulmat + Scale
- Jede Operation kostet eigene Allozierung und Memory Transfer

**Lösung:**
- Scale im Mulmat-Kernel integrieren (1 Kernel statt 2)
- `block_scores = ggml_mul_mat_scale(ctx0, k_blocks, q_f32, kq_scale)`
- CUDA Mulmat-Kernel können Skalierung inline durchführen

**Erwarteter Gewinn:**
- Zeit: -8% bis -15%
- Speicher: -1 Tensor (1 weniger temporäres Tensor)
- Implementierungsaufwand: Mittel (6-8h)

---

#### 5. Token-Score Batch-Ausführung (llama-graph.cpp:1964-1969)

**Aktuelle Situation:**
```cpp
// Einzelne Mulmat für jedes Query-Token
ggml_tensor * token_scores = ggml_mul_mat(ctx0, k_cand, q_f32);
token_scores = ggml_scale(ctx0, token_scores, kq_scale);
top_budget_indices = ggml_top_k(ctx0, token_scores, budget);
```

**Problem:**
- `k_cand` ist `[d, n_cand, n_head_kv, n_stream]`
- `q_f32` ist `[d, n_tokens, n_head, n_stream]`
- Mulmat broadcastet automatisch, aber ineffizient

**Lösung:**
- Top-K bereits im Mulmat integrieren (Fused Mulmat+TopK)
- Top-K könnte im CUDA-Kernel ohne extra Kernel-Aufruf erfolgen
- `q_f32` reshape oder split für batched Mulmat

**Erwarteter Gewinn:**
- Zeit: -10% bis -20%
- Implementierungsaufwand: Hoch (10-12h)

---

### **MITTEL** 🟢 - Geringere Wirkung, hoher Aufwand

#### 6. Optimierung von block_gather Indizes

**Aktuelle Situation:**
```cpp
// Im Kernel: Zugriff auf block_indices per Basis+offset
const int64_t idx_offset = im * idx_nb0 + 0 * 0 + ih_q * idx_nb2 + ib * idx_nb3;
const int32_t blk_idx = *(const int32_t *)((const char *)block_indices + idx_offset);
```

**Problem:**
- Jeder Thread berechnet seinen eigenen Offset
- Shared Memory könnte Block-Indizes cachen

**Lösung:**
- Block-Indizes in Shared Memory cachen
- Reduziert indirekten Speicherzugriff um 90%

**Erwarteter Gewinn:**
- Zeit: -3% bis -7%
- Implementierungsaufwand: Mittel (8-10h)

---

#### 7. Fused Kernel für Block Pool + Block Scatter

**Aktuelle Situation:**
- Block Pool (Kernel 1) erstellt Blocks
- Dann block_gather (Kernel 2) gathert Blöcke zurück

**Problem:**
- Zwischenspeicher alloziiert (k_blocks)
- Speichertransfer zwischen Kernel 1 und Kernel 2

**Lösung:**
- Fused Kernel: Block Pool + direkt aus Block-Pool gathern
- Eliminiert temporäres k_blocks Tensor

**Erwarteter Gewinn:**
- Zeit: -5% bis -10%
- Speicher: -50% temporärer Speicher
- Implementierungsaufwand: Hoch (12-16h)

---

#### 8. GQA Mapping im Block Pool Kernel

**Aktuelle Situation:**
- GQA Ratio berechnet in `build_hisa_sparse_attn` (line 179)
- Angewendet in block_gather (line 127)
- Muss redundant berechnet werden in jedem Kernel

**Lösung:**
- GQA Mapping als Op-Parameter übergeben
- Berechnung im CUDA-Kernel einmalig
- Vermeidet Redundanz

**Erwarteter Gewinn:**
- Zeit: -2% bis -4%
- Implementierungsaufwand: Niedrig (2-4h)

---

### **NIEDRIG** 🔵 - Geringe Wirkung, sehr hoher Aufwand

#### 9. Tensor-View statt allozieren

**Aktuelle Situation:**
- Viele `ggml_view` verwenden, aber auch `ggml_new_tensor_*`
- Fehlende Views reduzieren Allozierung

**Problem:**
- Manche Tensoren unnötig alloziert werden
- View-Infrastruktur ist vorhanden, aber nicht genutzt

**Lösung:**
- Views statt allozierung für temporäre Tensoren
- Reduziert Peak-Speicher um 20-30%

**Erwarteter Gewinn:**
- Speicher: -20% bis -30%
- Zeit: -3% bis -5% (weniger Speichermanagement)
- Implementierungsaufwand: Hoch (8-12h)

---

#### 10. Flash Attention KQ-Bias Unterstützung

**Aktuelle Situation:**
- HISA ignoriert kq_bias (setzt auf nullptr)
- Kommentiert in Build Mask Step (line 203)

**Problem:**
- kq_bias kann wichtig sein für certain attention patterns
- Wird ignoriert, könnte Qualität reduzieren

**Lösung:**
- kq_bias in HISA-Prozess integrieren
- Fused kernel für kq_bias + attention score

**Erwarteter Gewinn:**
- Qualität: ++ (kein qualitativer Verlust)
- Zeit: 0% bis +5% (weniger effizient)
- Implementierungsaufwand: Hoch (12-16h)

---

## Priorisierungsvorschlag

### **Phase 1: Schnelle Wins (< 1 Woche, > 20% Gesamtgewinn)**

1. ✅ **F32 Cast Eliminierung** (-15% bis -25% Zeit)
2. ✅ **KQ-Scale Integration** (-8% bis -15% Zeit)
3. ✅ **GQA Mapping Optimierung** (-2% bis -4% Zeit)

**Zusammenfassung:**
- Zeitgewinn: -25% bis -44%
- Aufwand: Niedrig bis Mittel
- Risiko: Niedrig

---

### **Phase 2: Mittelgroße Optimierungen (1-2 Wochen, 10-15% Gesamtgewinn)**

4. **Block-Score Scoring ohne Cast** (-10% bis -20% Zeit)
5. **Token-Score Batch-Ausführung** (-10% bis -20% Zeit)
6. **F32 Cast Eliminierung v2** (F16 block_pool statt F32, -15% bis -25% Zeit)

**Zusammenfassung:**
- Zeitgewinn: -10% bis -20%
- Aufwand: Mittel
- Risiko: Mittel

---

### **Phase 3: Advanced Optimierungen (2-4 Wochen, 5-10% Gesamtgewinn)**

7. **Fused Kernel für Block Pool + Scatter** (-5% bis -10% Zeit)
8. **Tensor-Contiguity Eliminierung** (-5% bis -10% Zeit)
9. **block_gather Shared Memory Cache** (-3% bis -7% Zeit)

**Zusammenfassung:**
- Zeitgewinn: -5% bis -27%
- Aufwand: Mittel bis Hoch
- Risiko: Mittel

---

## Mess- und Test-Rahmen

### Benchmark-Strategie

```bash
# Wikitext-2 Benchmark mit verschiedenen Parametern
llama-batched-bench -m model.gguf -p '...'   --hisa   --hisa-block-size 64   --hisa-top-m 8   --hisa-budget 1024   --hisa-min-tokens 2048   -n 100 -b 8 -t 8
```

### Metriken
- **Time per token**: Sekunden pro generiertem Token
- **Total time**: Gesamtdauer für Benchmark
- **Peak memory**: Maximale Speichernutzung während Benchmark
- **KV cache hit rate**: Wie oft wird HISA aktiviert?
- **Attention quality**: Perplexity Unterschied zu non-HISA

### Parameter-Grid
- **Block size**: 16, 32, 64, 128, 256
- **Top-M**: 1, 2, 4, 8, 16, 32
- **Budget**: 256, 512, 1024, 2048
- **Min tokens**: 1024, 2048, 4096, 8192

---

## Expected Total Performance Gains

### Konservative Schätzung (Phase 1+2)
- **Zeit**: -30% bis -50% gegenüber non-HISA
- **Speicher**: -20% bis -40%
- **Qualität**: < 1% Perplexity Unterschied

### Optimistische Schätzung (Phase 1+2+3)
- **Zeit**: -45% bis -65% gegenüber non-HISA
- **Speicher**: -30% bis -50%
- **Qualität**: < 2% Perplexity Unterschied

### Realistische Schätzung (Phase 1+2+3 + empirische Validierung)
- **Zeit**: -40% bis -60% gegenüber non-HISA
- **Speicher**: -25% bis -45%
- **Qualität**: < 1.5% Perplexity Unterschied

---

## Implementierungs-Risiken

### Niedriges Risiko
- F32 Cast Eliminierung
- KQ-Scale Integration
- GQA Mapping Optimierung

### Mittleres Risiko
- Token-Score Batch-Ausführung
- Tensor-Contiguity Eliminierung
- Fused Kernels

### Hohes Risiko
- Fused Block Pool + Scatter
- Shared Memory Cache in block_gather
- Tensor-View statt allozieren (komplexe Abhängigkeiten)

---

## Empfohlener Start

### Woche 1: F32 Cast Eliminierung + KQ-Scale Integration
**Warum:**
- Schnell implementierbar (4-6h total)
- Hoher ROI (20-30% Gesamtgewinn)
- Geringes Risiko
- Ermöglicht spätere Validierung

### Woche 2: Block-Score Scoping + F16 Block Pool
**Warum:**
- Nutzt Lessons Learned von Woche 1
- Weitere 15-25% Gewinn
- Erweitert Performance-Fundament

### Woche 3-4: Fused Kernels + Advanced Optimierungen
**Warum:**
- Jetzt sicher, Basis ist stabil
- Kann iterativ optimiert werden
- Kann nach Bedarf abgebrochen werden
