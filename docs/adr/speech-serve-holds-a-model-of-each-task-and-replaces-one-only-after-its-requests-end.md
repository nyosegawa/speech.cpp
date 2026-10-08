# speech serve holds a model of each task, and replaces one only after its requests end

## Context

A page or a program that both speaks and transcribes needs a model of each task, a transcription by the regions where
someone speaks needs a detection model beside the recognizer
([the record of transcription by regions](transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)),
and the page of `speech serve --open`
([the page's record](speech-serve-has-a-page-guarded-by-a-token-a-loopback-host-and-the-origin.md)) lets a user pick
models of the catalog and switch between them without restarting the server. A model takes 0.8 to 8 GB of memory, and
the machines speech.cpp runs on include laptops with 8 GB; Silero VAD takes 1.2 MB.

## Decision

- **The server holds at most one model of each task: synthesis, recognition and detection.** `speech serve` takes up to
  three models, one of each task, and `--open` starts it with none. `/v1/audio/speech` uses the synthesis model,
  `/v1/audio/transcriptions` the recognition model, and with `chunking_strategy` the detection model too, which no
  endpoint takes alone; `/v1/models` lists the models held in that order. A request to an endpoint without its model is
  a 404 that says which model the server holds, and one with `chunking_strategy` to a server without a detection model
  a 400 that says to give it one. No command chooses the detection model: it is given like the others.
- **The page replaces the model of a task by a model of the catalog.** While its file is fetched, the model held keeps
  serving. Then the server lets that model go, waits until the last request on it has ended, and only then loads the new
  one, so that two models of a task never take memory together; meanwhile the endpoint of that task answers a 503
  (`model_loading`) with `Retry-After`. A load that fails leaves the task without a model, and says why. A second load of
  a task while one runs is a 409.
- **The page loads a model of the catalog into the place of its task, detection included**, through the same endpoint;
  `/speech/models` gives the place of each task.
- **Each request keeps the models it runs on** until it has ended and its library requests have been freed, a
  transcription by regions its detection model as well, and each recognition and synthesis model keeps its own queue,
  so a transcription does not wait for a speech; a transcription by regions takes its turn on the recognition model.
- **`--add-voice` adds to the synthesis model given on the command line**, `--device`, `--threads` and `--no-warmup`
  apply to every model the server loads, the page's too, and a voice the page adds lasts while its model stays loaded.

The alternatives were turned down:

- Loading the new model before letting the old one go, so that requests never wait. It needs memory for both, which a
  laptop running a 2 GB model does not have.
- Cancelling the old model's requests at once. A speech the user is listening to would break off; the page stops its own
  requests before it replaces a model.
- Any number of models per task, chosen by the request's `model`. The memory of each would stay taken until the server
  ends, and nothing would decide which to let go.

## Consequences

A client that names the model in a request names the one of its task; one that leaves it out gets the model held. A
program that starts `speech serve` with two models gets both endpoints from one process.
