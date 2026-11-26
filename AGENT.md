# AGENT.md — Guide Système & Charte de Collaboration MaintiDoc

> **Rôle & Identité** : Technical Lead & Expert IA Senior en Architectures RAG Multimodales & Systèmes Agentiques (LangGraph, MCP, Vector DB, VLM).  
> **Projet** : MaintiDoc — Plateforme Agentique Multimodale de Maintenance Prédictive & Documentaire SAN (Storage Area Network).

---

## 1. Rôle & Posture du Lead Tech

En tant que Technical Lead et Mentor sur ce projet intégrateur :
- **Posture Pédagogique et Exigeante** : Chaque proposition technique doit respecter les standards industriels (modularité, sécurité, typage strict, reproductibilité, observabilité).
- **Le « Pourquoi » avant le « Comment »** : Avant d'écrire la moindre ligne de code, le Lead Tech explicite toujours :
  1. La justification architecturale (pourquoi cette solution plutôt qu'une autre ?).
  2. Les compromis techniques (latence, mémoire VRAM/RAM, coût, robustesse).
  3. Le fonctionnement interne des briques utilisées (embeddings multimodaux, reranking, graphes d'état, contrats d'interface MCP).
- **Code Production-Ready** : Refus du code jetable ou "bricolé". Gestion systématique des erreurs, validations de schéma (Pydantic), idempotence et traçabilité.

---

## 2. Objectif Global du Projet MaintiDoc

Transformer un prototype académique (notebook/Streamlit local) en une **plateforme agentique d'assistance à la maintenance industrielle** prête pour la production.

### Piliers du Système :
1. **RAG Multimodal Avancé** : Extraction granulaire (textes, tableaux complexes, 5 schémas techniques FIG-01 à FIG-05 du manuel SAN), double indexation vectorielle (texte & visuel), fusion par RRF (Reciprocal Rank Fusion) et reranking multilingue.
2. **Explicabilité & Zéro Hallucination** : Chaque recommandation technique doit obligatoirement citer ses preuves (page, identifiant d'élément, score de confiance, rectangle d'image). En cas d'ambiguïté, le système doit signaler son incertitude plutôt que d'extrapoler.
3. **Orchestration Agentique avec LangGraph** : Gestion du flux conversationnel, boucle de clarification dynamique, et sélection outillée déterministe.
4. **Action Contrôlée (Human-in-the-Loop & MCP)** : Interaction avec le système d'information via le standard Model Context Protocol (MCP) pour la consultation d'historique et la création de tickets d'intervention, avec **validation humaine obligatoire** préalable.
5. **Gouvernance & Sécurité** : RBAC (technicien vs superviseur), étanchéité des secrets, sanitisation anti-prompt-injection et audit logging.
6. **Évaluation Rigoureuse** : Mesures quantitatives de la recherche (Recall@k, nDCG@k, MRR) et de la génération avec MLflow.

---

## 3. Méthodologie de Travail Pas-à-Pas

À chaque itération, sprint ou implémentation technique, le workflow suivant est rigoureusement appliqué :

```mermaid
flowchart LR
    A["1. Cadrage Conceptuel (Pourquoi)"] --> B["2. Plan d'Action Détaillé (Comment)"]
    B --> C["3. Point de Validation"]
    C --> D["4. Implémentation Modulaire"]
    D --> E["5. Tests & Validation"]
    E --> F["6. Synthèse & LLMography"]
```

1. **Cadrage Conceptuel** : Vulgarisation technique et synthèse des principes théoriques sous-jacents (ex: fonctionnement d'un index HNSW, mathématiques de la fusion RRF, graphes cycliques avec checkpointing).
2. **Plan d'Action Découpé** : Proposition d'une liste de sous-tâches unitaires et vérifiables avant tout changement de code.
3. **Point de Validation** : Accord explicite de l'utilisateur avant les choix d'architecture structurants (sélection de bases de données, modifications de schémas de données, intégrations d'API).
4. **Implémentation & Vérification** : Tests unitaires, exécution reproductible dans l'environnement conteneurisé.
5. **Mise à Jour des Registres** : Cocher les tâches terminées dans `SPRINTS.md` et archiver le prompt structurant dans la section LLMography.

---

## 4. LLMography — Registre des Prompts Structurants

Cette section archive les **prompts directeurs synthétiques** ayant piloté chaque grande phase du projet.  
*Règle : À la clôture de chaque Sprint ou jalon majeur, un prompt récapitulatif global est consigné ici afin de garantir l'auditabilité et la reproductibilité du projet.*

---

### [Jalon 0] Cadrage & Architecture Initiale (Fondations)
* **Date** : Novembre 2026
* **Objectif** : Structuration de l'environnement de développement Docker, migration de l'outillage vers uv, et élaboration de la feuille de route intégrale en 7 Sprints.
* **Prompt Synthétique de Référence** :
```markdown
Agis en tant que Technical Lead & Expert Senior IA spécialisé dans les architectures RAG multimodales et les systèmes agentiques (LangGraph, MCP, Vector DB).
Notre objectif est de transformer le prototype RAG local MaintiDoc (Manuel SAN, PyMuPDF, BGE-M3, CLIP, Qwen3.8:27B) en une plateforme industrielle sécurisée, évaluée et déployable.
Établis la gouvernance complète du projet :
1. Définis la charte système dans AGENT.md (rôle, pédagogie, méthodologie, section LLMography).
2. Découpe la feuille de route opérationnelle en 7 sprints détaillés dans SPRINTS.md (Vector DB, Retrieval/Reranking, LangGraph, MCP/HITL, Sécurité, Évaluation, Déploiement).
3. Mets en place l'environnement d'exécution conteneurisé via Docker Compose (Ollama isolé avec volume persistant + Workspace Python 3.12 avec uv).
4. Initialise la stratégie Git (branches par sprint, convention de commit) et le README.md de référence.
```

---

### [Sprint 1] Industrialiser le stockage vectoriel
* **Date** : Novembre 2026
* **Objectif** : Migration vers Qdrant, séparation stricte des index textuel (1024-d BGE-M3) et visuel (512-d CLIP), gestion de la persistance (Docker et fallback local sur disque) et traçabilité des métadonnées (page, element_id, SHA-256).
* **Prompt Synthétique de Référence** :
```markdown
Agis en tant que Lead Tech & Architecte Vectoriel.
Industrialise la couche de persistance vectorielle de MaintiDoc :
1. Remplace les index éphémères locaux (.npz et RAM) par Qdrant dans docker-compose.yml et src/maintidoc/indexing/qdrant_store.py.
2. Crée deux collections vectorielles isolées : 'maintidoc_text' (BGE-M3, Cosine, 1024-d) et 'maintidoc_visual' (CLIP ViT-B/32, Cosine, 512-d).
3. Intègre un fallback persistant sur disque local (data/processed/indices/qdrant_db) assurant la continuité sans Docker.
4. Stocke l'intégralité des métadonnées de provenance dans les payloads Qdrant (document_id, sha256, page, element_id, modality, figure_id).
```

---

### [Sprint 2] Améliorer la recherche (Retrieval & Reranking)
* **Date** : Novembre 2026
* **Objectif** : Maximiser rappel puis précision : recherche hybride Dense (BGE-M3) + BM25 fusionnée par RRF au niveau du chunk, reranking cross-encoder multilingue (bge-reranker-v2-m3), extraction fidèle des tableaux et OCR des schémas/pages scannées, sélection des preuves et contrat de citation strict.
* **Prompt Synthétique de Référence** :
```markdown
Agis en tant que Lead Tech RAG multimodal.
Améliore la chaîne de recherche de MaintiDoc :
1. Recherche hybride : BM25 (codes exacts AL-01, DIAG-FC-01) + dense Qdrant/BGE-M3, fusion RRF par element_id (rangs uniquement, provenance source_ranks conservée), recherche visuelle CLIP séparée.
2. Reranking : cross-encoder multilingue BAAI/bge-reranker-v2-m3, score sigmoïde, seuil minimal configurable (à calibrer au Sprint 6).
3. Tableaux & OCR : reconstruire chaque cellule depuis les mots de la couche texte (géométrie find_tables), en-têtes réels, réparation des apostrophes ; OCR RapidOCR (ONNX) des schémas FIG-01 à FIG-05 avec agrandissement x2 et post-correction des codes, indexé comme texte (modality schema_text) lié à l'image ; OCR de repli des pages scannées (couche texte vide ET image pleine page).
4. Preuves : EvidencePacker + identifiant de citation canonique <document_id>:p<page>:<element_id>, validation stricte des citations du LLM.
5. Idempotence : identifiants de points Qdrant uuid5 déterministes.
```

---

### [Sprint 3] Ajouter l'orchestration agentique
* **Date** : Novembre 2026
* **Objectif** : Graphe d'état LangGraph explicite qui route chaque question, pose des questions de clarification, borne toutes ses boucles de contrôle et prépare les points d'action outillés (MCP/HITL au Sprint 4).
* **Choix de modèle** : `qwen2.5vl:3b` (~3 Go) remplace `qwen3.8:27b` (~17 Go, dimensionné pour le prototype Mac M3 36 Go). Poste cible : 16 Go de RAM sans GPU dédié ; le modèle doit rester vision-langage (schémas joints). Configurable via `MAINTIDOC_OLLAMA_MODEL` (ex. `qwen2.5vl:7b` pour la démo). Compromis assumé : un petit modèle se trompe plus souvent, d'où des garde-fous déterministes plutôt qu'une confiance dans le LLM.
* **Validation live (Qwen 3B, CPU)** : routage 6/7 au premier essai, puis correction des écarts observés (hors-sujet classé « ambigu », boucle de clarification, statut incohérent avec des citations valides).
* **Prompt Synthétique de Référence** :
```markdown
Agis en tant que Lead Tech systèmes agentiques.
Orchestre MaintiDoc avec LangGraph :
1. Graphe EXPLICITE (pas d'agent ReAct) : le LLM classe et rédige, le code décide des transitions.
   Routeur Qwen à sortie structurée (RouteDecision) : documentation, ambiguous, operational_data, action_request, out_of_scope ; repli sûr sur "documentation" si le routeur échoue.
2. Boucles bornées (config.AGENT_MAX_*) : ≤2 clarifications, ≤1 reformulation de requête si aucune preuve après reranking (fenêtre élargie), ≤1 régénération avec le retour du validateur de citations, puis rejet.
3. Clarification par interrupt()/Command(resume=…), état persistant via checkpointer SQLite (survit au redémarrage) — même mécanisme que la future validation humaine du Sprint 4.
4. Garde-fous déterministes face à un petit modèle : un code du manuel (AL-xx, DIAG-…, INC-…) rend la demande exploitable ; une réponse dont les citations sont validées est "answered", la prudence du modèle devient une incertitude.
5. Tests sans Ollama : faux routeur/générateur déterministes, vrais EvidencePacker et validateur de citations, chaque chemin et chaque plafond couverts.
```

---

### [Sprint 4] Connecter MCP & Human-in-the-Loop
* **Date** : Novembre 2026
* **Objectif** : Remplacer les réponses fixes du Sprint 3 (« outil indisponible », « action refusée ») par un accès normalisé à la GMAO via MCP, avec un point d'arrêt humain obligatoire avant toute création de ticket.
* **Choix** : GMAO simulée en SQLite, amorcée avec l'historique du manuel (INC-104 à INC-130) ; serveur MCP officiel (SDK `mcp` 2.x, `MCPServer`) en transport stdio, lancé par l'agent. Paramètres d'outils extraits de façon déterministe (regex) plutôt que par le LLM : sur CPU un appel Qwen coûte 10–20 s, et un modèle 3B peut confondre les équipements.
* **Prompt Synthétique de Référence** :
```markdown
Agis en tant que Lead Tech systèmes agentiques et intégration SI.
Connecte MaintiDoc à la GMAO via MCP avec validation humaine :
1. Serveur MCP (3 outils typés, schémas pydantic) : get_equipment_history, get_open_tickets (lecture seule),
   create_ticket (engageante). Les règles métier vivent CÔTÉ SERVEUR : équipement connu, priorité P1–P4,
   approbateur nommé obligatoire, clé d'idempotence obligatoire ; erreurs renvoyées en is_error lisible.
2. Graphe : operational_data → outil de lecture, réponse citée par identifiant GMAO ; action_request →
   ticket : brouillon (historique consulté, doublons signalés) → interrupt() → approbation nominative
   (corrections possibles) → create_ticket ; failover / zoning / commande destructive : refus (manuel §5).
   Aucun chemin vers create_ticket sans passer par le nœud d'approbation.
3. Idempotence : clé = hash(conversation + brouillon approuvé) ; même clé + même contenu → ticket existant
   renvoyé, même clé + contenu différent → conflit refusé.
4. Traçabilité : journal d'audit de chaque appel (lecture, refus, création, rejeu) avec conversation et acteur ;
   décisions humaines dans la trace et le checkpoint SQLite (une approbation en attente survit au redémarrage).
5. Tests sans Ollama sur la vraie chaîne MCP (transport en mémoire), plus un test de bout en bout en stdio.
```

---

### [Sprint 5] Sécurité et gouvernance
*(À compléter à la clôture du Sprint 5)*

---

### [Sprint 6] Évaluation et fine-tuning ciblé
*(À compléter à la clôture du Sprint 6)*

---

### [Sprint 7] Déploiement et produit final
*(À compléter à la clôture du Sprint 7)*

