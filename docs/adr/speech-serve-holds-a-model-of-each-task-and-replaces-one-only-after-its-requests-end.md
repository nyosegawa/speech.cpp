# speech serve holds a model of each task, and replaces one only after its requests end

## Context

A page or a program that both speaks and transcribes needs a model of each task, and the page of `speech serve --open`
([the page's record](speech-serve-has-a-page-guarded-by-a-token-a-loopback-host-and-the-origin.md)) lets a user pick
models of the catalog and switch between them without restarting the server. A model takes 0.8 to 8 GB of memory, and
the machines speech.cpp runs on include laptops with 8 GB.

## Decision

- **The server holds at most one synthesis model and one recognition model.** `speech serve` takes up to two models, one
  of each task, and `--open` starts it with none. `/v1/audio/speech` uses the synthesis model and
  `/v1/audio/transcriptions` the recognition model; `/v1/models` lists both. A request to a task without a model is a
  404 that says which model the server holds. A model of detection, which no endpoint takes, is refused when the server
  starts, and the page loads none.
- **The page replaces the model of a task by a model of the catalog.** While its file is fetched, the model held keeps
  serving. Then the server lets that model go, waits until the last request on it has ended, and only then loads the new
  one, so that two models of a task never take memory together; meanwhile the endpoint of that task answers a 503
  (`model_loading`) with `Retry-After`. A load that fails leaves the task without a model, and says why. A second load of
  a task while one runs is a 409.
- **Each request keeps the model it runs on** until it has ended and its library request has been freed, and each model
  keeps its own queue, so a transcription does not wait for a speech.
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
