# Segmentación de instancias celulares y clasificación de texto

**Investigación empírica reproducible · Tarea 3 (Sección 4.8) · Deep Learning · Maestría en Ingeniería Biomédica,
Universidad del Norte · Prof. Dr. Lihki Rubio**

Autor: Andrés Duarte Chaves

## Resumen

```{include} resumen.md
```

## Introducción

Este trabajo estudia dos problemas con el **mismo estándar metodológico**, siguiendo el enunciado de la sección
4.8 de la guía del curso:

1. **Segmentación de instancias celulares** en microscopía de contraste de fase (Sartorius Cell Instance
   Segmentation; neuroblastoma SH-SY5Y, astrocitos y neuronas corticales). Contar y delimitar células una a una es
   la base de ensayos de neurodegeneración y de toxicidad, y en contraste de fase es difícil porque las células son
   tenues, se tocan y tienen morfologías muy distintas.
2. **Clasificación de texto** de hojas de vida en 24 categorías de cargo (Jarvis Calling Hiring Contest), un
   contexto de selección de personal en el que el análisis de equidad y de riesgo es obligatorio.

El objetivo no es solo saber qué modelo obtiene la mejor cifra, sino **cuándo, por qué, con qué incertidumbre, a qué
costo y con qué riesgos** funciona cada uno. Por eso el estudio: (i) pre-registra hipótesis y protocolo antes de
entrenar (capítulo 0, etiqueta git `prereg`); (ii) audita los datos antes de modelar (capítulos 3 y 6); (iii)
compara modelos con presupuesto de búsqueda equitativo, varias semillas e inferencia estadística sobre unidades de
análisis (imágenes o documentos), nunca sobre píxeles; (iv) aísla con ablaciones qué decisiones explican el
rendimiento; (v) mide robustez, calibración, incertidumbre, explicabilidad y equidad; y (vi) despliega el modelo
elegido de cada dominio como un servicio probado, versionado y monitoreado, con un experimento de detección de
deriva.

## Cómo leer este libro

| Parte | Capítulos |
|---|---|
| Métodos | 0 · Pre-registro · 1 · Fundamentos teóricos · 2 · Verificación de los ejercicios manuales (LSTM y word2vec) |
| Bloque 1 · Imágenes | 3 · EDA y auditoría · 4 · Modelos y evaluación · 5 · Ablaciones, robustez, calibración y eficiencia |
| Bloque 2 · Texto | 6 · EDA y auditoría · 7 · Modelos y evaluación · 8 · Robustez, calibración, explicabilidad y equidad |
| MLOps y discusión | 9 · Exportación, API, carga y deriva · 10 · Decisión multicriterio, hipótesis, revisión crítica · Conclusiones |

Toda cifra se reporta como media [IC 95 %]. Los notebooks se ejecutaron de arriba abajo en Google Colab con los
datos reales (perfil `full`); el código, las pruebas, la API, el tablero y las instrucciones de reproducción están
en el [repositorio](https://github.com/aduartech81/deep-learning-03-cellseg-nlp).

```{note}
Los dos ejercicios manuales de la sección 4.8 (LSTM con BPTT y word2vec) se resuelven a mano y se entregan por
separado, como exige la guía. El capítulo 2 solo verifica numéricamente esa solución.
```
