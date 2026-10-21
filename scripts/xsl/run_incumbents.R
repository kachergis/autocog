# Bridge for src/xsl/incumbents.py: run XSLmodels models on arbitrary designs.
#
#   Rscript scripts/xsl/run_incumbents.R <request.json> <response.json> <Rlib>
#
# request:  {"models":  [{"name", "constructor", "params": {...}, "fixed": {...}}],
#            "designs": [{"id", "train": {"words": [[...]], "objects": [[...]]},
#                         "test": null | {"words": [[w]], "objects": [[...]], "correct": [[o]]}}],
#            "n_sim": 500, "seed": 1}
# response: {"predictions": {<model name>: {<design id>: [per-item accuracy]}}}
#
# <Rlib> is the library holding the pinned XSLmodels build (normally
# xsl-bakeoff/fits/v05/Rlib), so predictions come from the same package
# version as the incumbent fits.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
request_path <- args[[1]]
response_path <- args[[2]]
rlib <- args[[3]]

.libPaths(c(rlib, .libPaths()))
suppressMessages(library(XSLmodels, lib.loc = rlib))

req <- jsonlite::read_json(request_path, simplifyVector = FALSE)

trials <- function(x) lapply(x, unlist)

designs <- lapply(req$designs, function(d) {
  test <- list()
  if (!is.null(d$test)) {
    test <- list(words = trials(d$test$words), objects = trials(d$test$objects))
    if (!is.null(d$test$correct)) test$correct <- trials(d$test$correct)
  }
  xslData(train = list(words = trials(d$train$words), objects = trials(d$train$objects)),
          test = test, label = as.character(d$id))
})
ids <- vapply(req$designs, \(d) as.character(d$id), character(1))

n_sim <- if (is.null(req$n_sim)) 500 else req$n_sim
seed <- if (is.null(req$seed)) 1 else req$seed

predictions <- list()
for (m in req$models) {
  fn <- get(m$constructor, envir = asNamespace("XSLmodels"))
  mod <- do.call(fn, c(m$params, m$fixed))
  set.seed(seed)
  res <- xsl_run(mod, designs, control = xslControl(n_sim = n_sim))
  predictions[[m$name]] <- setNames(lapply(res$fits, \(f) unname(as.numeric(f$perf))), ids)
}

jsonlite::write_json(list(predictions = predictions), response_path,
                     auto_unbox = FALSE, digits = NA, na = "null")
