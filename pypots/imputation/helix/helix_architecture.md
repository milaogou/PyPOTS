# Hybrid Temporal-Feature Encoding Architecture for Time Series Imputation

```mermaid
flowchart TB
    subgraph Input["<b>Input</b>"]
        X["X ∈ ℝ<sup>B×T×F</sup><br/>Time Series Data"]
        M["M ∈ {0,1}<sup>B×T×F</sup><br/>Missing Mask"]
    end

    subgraph Embedding["<b>Time Series Embedding Layer</b>"]
        direction LR
        V["Value<br/>x<sub>i,j</sub>"]
        RoPE["Rotary PE<br/>p<sub>t</sub> ∈ ℝ<sup>d<sub>pe</sub></sup>"]
        FID["Feature ID<br/>f<sub>j</sub> ∈ ℝ<sup>d<sub>f</sub></sup>"]
        MF["Mask<br/>m<sub>i,j</sub>"]
    end

    subgraph Concat["<b>Concatenation</b>"]
        E["E = [x || p<sub>t</sub> || f<sub>j</sub> || m]<br/>E ∈ ℝ<sup>B×T×F×d<sub>e</sub></sup>"]
    end

    subgraph Proj["<b>Linear Projection</b>"]
        H0["H<sup>(0)</sup> = W<sub>proj</sub>E<br/>H<sup>(0)</sup> ∈ ℝ<sup>B×T×F×d</sup>"]
    end

    subgraph Layer["<b>Hybrid Encoding Layer ×L</b>"]
        direction TB
        
        subgraph P1["Phase 1: Parallel Encoding"]
            direction LR
            TA["Temporal<br/>Attention<br/>A<sub>T</sub>(H)"]
            FA["Feature<br/>Attention<br/>A<sub>F</sub>(H)"]
        end
        
        subgraph P2["Phase 2: Cross-Dimensional Serial Encoding"]
            direction LR
            TF["A<sub>F</sub>(A<sub>T</sub>(H))<br/>Time→Feature"]
            FT["A<sub>T</sub>(A<sub>F</sub>(H))<br/>Feature→Time"]
        end
        
        subgraph Fusion["Intra-Layer Fusion"]
            AVG["H<sup>(l)</sup> = Mean(H<sub>T</sub>, H<sub>F</sub>, H<sub>TF</sub>, H<sub>FT</sub>)"]
        end
    end

    subgraph Global["<b>Global Multi-Level Fusion</b>"]
        GF["H̃ = Mean(H<sup>(0)</sup>, H<sup>(1)</sup>, ..., H<sup>(L)</sup>)"]
        LN["LayerNorm"]
    end

    subgraph Output["<b>Output Layer</b>"]
        OP["X̂ = W<sub>out</sub>H̃<br/>X̂ ∈ ℝ<sup>B×T×F</sup>"]
        Final["X̃ = M ⊙ X + (1-M) ⊙ X̂"]
    end

    X --> Embedding
    M --> Embedding
    V --> E
    RoPE --> E
    FID --> E
    MF --> E
    E --> H0
    H0 --> Layer
    H0 -.->|"skip"| GF
    
    Layer --> P1
    TA --> P2
    FA --> P2
    P2 --> Fusion
    Fusion -->|"iterate L times"| Layer
    Fusion -.->|"intermediate outputs"| GF
    
    GF --> LN
    LN --> OP
    OP --> Final

    style Input fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style Embedding fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style Concat fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    style Proj fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style Layer fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style P1 fill:#FCE4EC,stroke:#AD1457,stroke-width:1px
    style P2 fill:#FCE4EC,stroke:#AD1457,stroke-width:1px
    style Fusion fill:#FCE4EC,stroke:#AD1457,stroke-width:1px
    style Global fill:#E0F7FA,stroke:#00838F,stroke-width:2px
    style Output fill:#F1F8E9,stroke:#558B2F,stroke-width:2px
