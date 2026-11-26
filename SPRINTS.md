# SPRINTS.md — Feuille de Route Opérationnelle MaintiDoc

Suivi d'avancement des 7 Sprints du projet intégrateur MaintiDoc.  
*Chaque tâche fait l'objet d'un cadrage technique préalable, d'une validation par le Lead Tech et d'un commit dédié.*

---

## Sprint 1 : Industrialiser le stockage vectoriel
> **Objectif** : Remplacer les index locaux éphémères (`npz` / stockage en mémoire) par une base de données vectorielle d'entreprise persistante et optimisée pour la recherche hybride.

- [x] Migration vers PGVector ou Qdrant (remplacement des index locaux)
- [x] Séparation explicite des index textuels et visuels
- [x] Gestion des métadonnées, de la provenance et de la persistance des données

---

## Sprint 2 : Améliorer la recherche (Retrieval & Reranking)
> **Objectif** : Maximiser le rappel et la précision du système d'extraction en couplant recherche dense/sparse, alignement multimodal et reclassement par cross-encoder.

- [x] Implémentation de la recherche hybride (texte + image)
- [x] Intégration d'un reranker multilingue
- [x] Traitement avancé des tableaux et de l'OCR
- [x] Mécanisme de sélection des preuves et validation stricte des citations

---

## Sprint 3 : Ajouter l'orchestration agentique
> **Objectif** : Structurer l'intelligence du système via un graphe d'état avec LangChain & LangGraph pour router, poser des questions de clarification et décider des actions outillées.

- [x] Mise en place du graphe d'état avec LangChain & LangGraph
- [x] Définition des boucles de contrôle et gestion des états/transitions
- [x] Gestion des demandes de clarification utilisateur et stratégies d'outils

---

## Sprint 4 : Connecter MCP & Human-in-the-Loop
> **Objectif** : Normaliser l'accès aux données opérationnelles via le protocole MCP et imposer un point d'arrêt humain obligatoire pour toute action engageante (création de tickets).

- [x] Intégration du protocole MCP (outils : `get_equipment_history`, `get_open_tickets`, `create_ticket`)
- [x] Implémentation du workflow de validation humaine obligatoire avant création de ticket
- [x] Assurer l'idempotence et la traçabilité des actions

---

## Sprint 5 : Sécurité et gouvernance
> **Objectif** : Protéger l'application contre les compromissions, restreindre les accès aux documents sensibles selon le profil utilisateur et journaliser l'activité.

- [ ] Authentification, Autorisation & RBAC
- [ ] Contrôle d'accès documentaire par rôle
- [ ] Protections contre le Prompt Injection et gestion sécurisée des secrets (.env)
- [ ] Journalisation (logging) minimale des opérations

---

## Sprint 6 : Évaluation et fine-tuning ciblé
> **Objectif** : Quantifier scientifiquement les performances du pipeline RAG et de l'agent sur un jeu d'évaluation métier représentatif avec MLflow.

- [ ] Création d'un jeu de test multi-scénarios
- [ ] Calcul des métriques de recherche (Recall@5, nDCG@5, MRR@10)
- [ ] Évaluation de l'exactitude des réponses et de la qualité des citations
- [ ] Fine-tuning/adaptation du reranker et analyse des erreurs

---

## Sprint 7 : Déploiement et produit final
> **Objectif** : Finaliser l'infrastructure de production conteneurisée, connecter les canaux de communication client et livrer les livrables du projet intégrateur.

- [ ] Conteneurisation complète avec Docker Compose
- [ ] Monitoring et observabilité avec MLflow
- [ ] Finalisation de l'application web et intégration client (Telegram / Discord)
- [ ] Rédaction de la documentation finale et préparation du support de soutenance

