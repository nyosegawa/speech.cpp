# Haru takes its character card's fields as options and writes the card in the request's language

## Context

Haru's prompt begins with a character card, a ChatML system turn that the model was trained to read in one layout: a
lead sentence and, for each field that has a value, a header and the value, in one of nine languages. The official
server and app take the fields (persona, name, gender, backchannel, other_name, relationship, situation) and render the
card with the wording of the chosen language (`card_templates.json`); the official `DuplexSession` takes the rendered text
as `system_prompt`. A gender or backchannel outside the template's choices is left out of the card by the official code.

## Decision

- **The card's fields are request options** under the official card's names, `persona`, `name`, `gender`,
  `backchannel`, `other_name`, `relationship` and `situation`, strings, with `gender` and `backchannel` taking the
  choices the model file lists. The page and the tools build their forms from the options the model declares, as for
  every other model.
- **The model file holds the card's wording** in each of its languages, and the library renders the card in the
  request's `language`, which a card needs: a field set with the language `auto` is `invalid_argument` naming
  `language`. A conversation with no field set has no card.
- **A choice the card does not offer is refused** with the option named, where the official code leaves it out of the
  card.
- The sampling, the guidance and the start of speech take the official session's names, `temperature`, `top_k`,
  `top_p`, `guidance_scale`, `eop_bias` and `speak_first`, OpenAI's audio API having none of them; the length is
  `max_seconds`. DRY, which the official session always applies, is not an option.

The alternatives were turned down:

- The whole card as one text, under OpenAI's `instructions`. Every caller would hold the card's wording in nine languages
  and its layout, which the model file holds once, and a text in another layout is not what the model was trained on.
- Both, the text and the fields. Two inputs for one card, which a request could set at odds.
- Leaving an unknown choice out, as the official code does. The caller asked for something the model does not get.
