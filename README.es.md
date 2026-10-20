# Motor de contexto para aplicaciones con LLMs

Este proyecto decide qué información llega al modelo cuando una conversación crece:
política estable, restricciones vigentes, turnos recientes completos y un archivo
histórico que se reduce por capas. Puedes inspeccionar cada decisión y su procedencia.

El problema práctico: guardar solo los últimos mensajes puede perder una restricción
importante; enviar todo consume contexto y mantiene valores antiguos. El motor conserva
un estado explícito que tu aplicación actualiza con revisiones y comprobaciones de versión.
No extrae automáticamente hechos de texto ni ejecuta herramientas.

```sh
git clone https://github.com/sn41lz/agent-systems-context-window.git
cd agent-systems-context-window
make verify
```

Necesitas Docker con Compose v2. Se preparan las dependencias, se ejecutan las pruebas
y se genera `artifacts/demo/report.html`. La ejecución es local y sin red; la primera
construcción descarga dependencias. Con Python 3.12–3.14 puedes usar `make local`.

En la referencia de 120 turnos, el historial completo termina en 25.968 tokens de
aplicación; la política por capas, en 2.860, conservando los seis valores vigentes.
Son mediciones locales sobre una conversación sintética: no demuestran precisión de
respuestas, ahorro facturado ni latencia del proveedor.

La adaptación a las APIs actuales incluye controles opcionales de caché, costes de
lectura y escritura, reserva de salida para razonamiento, recuento de tokens del
proveedor y una prueba separada de compactación nativa que conserva su estado opaco.
Los modelos y precios se configuran explícitamente. Solo hay adaptador live de OpenAI.

También puedes compilar tu propia conversación sin llamar a un modelo:

```sh
make setup
.venv/bin/context-window compile \
  --transcript examples/incident.jsonl \
  --query "¿Qué restricciones siguen vigentes?"
```

La salida incluye una petición JSON lista para inspeccionar, metadatos y trazas sin
contenido de mensajes. El archivo de petición sí contiene tu contexto y se guarda con
permisos privados. Las llamadas live requieren clave, pueden generar gastos y omiten
las respuestas guardadas salvo que actives `--include-content`.

El formato manual de turnos no sirve para reconstruir arbitrariamente estados de
razonamiento de un proveedor. El motor es local y en memoria. La compactación puede
perder información; si los elementos protegidos no caben, falla explícitamente.

[Instrucciones completas, resultados y límites](README.md) ·
[Arquitectura](docs/architecture.md) · [APIs actuales](docs/current-models.md) ·
[Licencia MIT](LICENSE)
