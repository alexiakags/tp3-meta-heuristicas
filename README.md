# Trabalho Prático 3 – Meta-heurísticas (CCF-480)

Implementação de uma meta-heurística cooperativa (PSO cooperativo) para:

- **Problema 1 do TP1 (configuração a)**;
- **Problema 1 do TP2** com tratamento de restrições por penalidade quadrática.

## Como executar

```bash
python run_experiments.py
```

Ao executar, o código realiza **30 execuções independentes por problema** e gera:

- `results/summary.json` com:
  - mínimo;
  - máximo;
  - média;
  - desvio padrão;
  - melhor solução (valor e variáveis de decisão);
  - histórico de todas as 30 execuções.
- `results/boxplots/*.svg` com os boxplots de cada problema.

## Resultados obtidos (30 execuções)

### TP1 – Problema 1 (configuração a)

- **mínimo:** `9.183502657715032e-27`
- **máximo:** `4.7008119381848094e-23`
- **média:** `7.07135968004484e-24`
- **desvio padrão:** `1.3499294882149392e-23`
- **melhor solução (x):** `[1.5000000000000617, -1.9999999999999851, 0.4999999999999282]`
- **melhor valor:** `9.183502657715032e-27`

### TP2 – Problema 1 (com restrição)

- **mínimo:** `0.31993046908452527`
- **máximo:** `0.4468491554728149`
- **média:** `0.3678933903594976`
- **desvio padrão:** `0.04077040619673003`
- **melhor solução (x):** `[1.611607246398222, 0.5888047446356489]`
- **melhor valor:** `0.31993046908452527`