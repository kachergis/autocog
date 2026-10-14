# Export the XSLmodels corpus, the scorable holdouts, the xsl-bakeoff v05
# incumbent fits, and R reference predictions to JSON under data/xsl/, so the
# Python side (src/xsl/) never needs R to load data or to check its ports.
#
# Uses the XSLmodels build pinned for the v05 bakeoff fits
# (<bakeoff>/fits/v05/Rlib, commit in <bakeoff>/fits/v05/XSLMODELS_COMMIT), so
# every number exported here comes from the same package version as the fits.
#
# Run from the autocog repo root:
#   Rscript scripts/xsl/export_corpus.R [--bakeoff ../xsl-bakeoff] [--out data/xsl]
#
# Outputs:
#   corpus.json                 the 47 xsl_datasets conditions
#   holdout.json                the scorable holdout sets (sealed: evaluation only)
#   incumbents.json             v05 group fits for every model with a plain
#                               package constructor, plus v05's per-item ModelPerf
#   reference_predictions.json  xsl_run() perf + final matrix for the models
#                               ported to Python (parity fixtures)
#   manifest.json               provenance: commit, versions, md5 of inputs/outputs

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[[i + 1]]
}
bakeoff <- normalizePath(arg_value("--bakeoff", "../xsl-bakeoff"), mustWork = TRUE)
out_dir <- arg_value("--out", "data/xsl")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pinned_lib <- file.path(bakeoff, "fits", "v05", "Rlib")
stopifnot(dir.exists(file.path(pinned_lib, "XSLmodels")))
.libPaths(c(pinned_lib, .libPaths()))
suppressMessages(library(XSLmodels, lib.loc = pinned_lib))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

xslmodels_commit <- trimws(readLines(file.path(bakeoff, "fits", "v05", "XSLMODELS_COMMIT")))
fits_path <- file.path(bakeoff, "fits", "v05", "adults", "group_fits.Rdata")
fits_env <- new.env()
load(fits_path, envir = fits_env)

ub <- jsonlite::unbox
# A named list of scalars as a JSON object; `{}` (not `[]`) when empty.
obj <- function(x) if (length(x) == 0) setNames(list(), character(0)) else lapply(x, ub)

# Training/test trials stay arrays even when a trial has a single word, so
# Python always sees list-of-lists (auto_unbox is off for the whole file).
# Some conditions (e.g. 301) store a trial as a one-row data.frame, which
# jsonlite would write as a nested array; `unlist` flattens it exactly the way
# the models do (`tr_w <- unlist(data$words[t])`).
flat_trials <- function(trials) lapply(trials, unlist, use.names = FALSE)

export_condition <- function(d, index) {
  rm <- d$response_matrix
  test <- if (length(d$test) > 0) {
    t <- list(words = flat_trials(d$test$words), objects = flat_trials(d$test$objects))
    if (!is.null(d$test$correct)) t$correct <- flat_trials(d$test$correct)
    t
  } else NULL
  list(
    index = ub(index),
    label = ub(as.character(d$label)),
    condition = ub(as.character(d$condition)),
    description = ub(if (length(d$description) == 0) "" else as.character(d$description)),
    n_subj = ub(if (length(d$n_subj) == 0) NA else as.numeric(d$n_subj)),
    train = list(words = flat_trials(d$train$words), objects = flat_trials(d$train$objects)),
    test = test,
    accuracy = as.numeric(d$accuracy),
    response_matrix = if (is.null(rm)) NULL else list(
      items = rownames(rm), objects = colnames(rm), counts = unname(rm)
    )
  )
}

write_json <- function(x, name) {
  path <- file.path(out_dir, name)
  jsonlite::write_json(x, path, auto_unbox = FALSE, digits = NA,
                       null = "null", na = "null", pretty = FALSE)
  path
}

# ---- corpus + holdouts ------------------------------------------------------

corpus <- lapply(seq_along(xsl_datasets), \(i) export_condition(xsl_datasets[[i]], i))

holdout_meta <- xsl_holdout_datasets
scorable <- names(holdout_meta)[vapply(holdout_meta, `[[`, logical(1), "scorable")]
holdout <- lapply(scorable, \(nm) list(
  name = ub(nm),
  scoring = ub(holdout_meta[[nm]]$scoring),
  note = ub(holdout_meta[[nm]]$note),
  conditions = lapply(seq_along(get(nm)), \(i) export_condition(get(nm)[[i]], i))
))

# ---- incumbents ---------------------------------------------------------------

# v05 roster entries that map onto a plain package constructor (see
# <bakeoff>/fits/v05/specs.R). Parameter names are in DEoptim order, which is
# how bestmem is laid out. strength, rescorla-wagner and fazlyt are custom
# xslMod wrappers in specs.R and are deliberately not exported.
SPECS <- list(
  baseline           = list(constructor = "baseline", params = character(0)),
  kachergis          = list(constructor = "uncfam", params = c("X", "B", "C"), fixed = list(variant = "entropy")),
  novelty            = list(constructor = "uncfam", params = c("X", "B", "C"), fixed = list(variant = "novelty")),
  uncertainty        = list(constructor = "uncfam", params = c("X", "B", "C"), fixed = list(variant = "uncertainty-only")),
  Bayesian_decay     = list(constructor = "bayesian_decay", params = c("alpha", "delta", "ch_dec")),
  fazly              = list(constructor = "fazly", params = c("lambda", "beta")),
  uncfam_attention   = list(constructor = "uncfam_attention", params = c("X", "B", "C")),
  uncfam_predictive  = list(constructor = "uncfam_predictive", params = c("X", "B", "C")),
  kalman_filter      = list(constructor = "kalman_filter", params = c("tau2", "sigma2_obs", "sigma2_0")),
  uncfam_gamma       = list(constructor = "uncfam_gamma", params = c("X", "B", "C", "gamma")),
  uncfam_elimination = list(constructor = "uncfam_elimination", params = c("X", "B", "C", "eps")),
  tilles             = list(constructor = "tilles", params = c("x", "b", "alpha_0")),
  trueswell2012      = list(constructor = "propose_but_verify", params = c("alpha", "alpha_increase")),
  `guess-and-test`   = list(constructor = "guess_and_test", params = c("f", "sa")),
  pursuit            = list(constructor = "pursuit", params = c("gamma", "threshold", "lambda")),
  softmax_rl         = list(constructor = "softmax_rl", params = c("alpha", "beta")),
  minerva2           = list(constructor = "minerva2", params = "X", fixed = list(D = 100)),
  kachergis_sampling = list(constructor = "uncfam_sampling", params = c("X", "B", "C"), fixed = list(K = 1))
)

build_model <- function(spec, params) {
  fn <- get(spec$constructor, envir = asNamespace("XSLmodels"))
  do.call(fn, c(as.list(params), spec$fixed))
}

gfd <- fits_env$gfd
group_fits <- fits_env$group_fits
incumbents <- list()
skipped <- character(0)
for (name in names(SPECS)) {
  fit <- group_fits[[name]]
  if (is.null(fit)) { skipped <- c(skipped, name); next }
  spec <- SPECS[[name]]
  bestmem <- unname(fit$optim$bestmem)
  stopifnot(length(bestmem) == length(spec$params))
  params <- setNames(as.list(bestmem), spec$params)
  mod <- build_model(spec, params)
  rows <- gfd[gfd$Model == name, ]
  # v05's own per-item predictions, grouped by condition in corpus order; the
  # round-trip test compares the Python->Rscript bridge against these.
  v05_perf <- lapply(xsl_datasets, \(d) as.numeric(rows$ModelPerf[rows$Condition == d$condition &
                                                                     rows$condnum == d$label]))
  incumbents[[name]] <- list(
    name = ub(name),
    constructor = ub(spec$constructor),
    params = obj(params),
    fixed = obj(spec$fixed),
    stochastic = ub(isTRUE(mod$stochastic)),
    sse_v05 = ub(fit$optim$bestval),
    n_params = ub(length(spec$params)),
    v05_perf = v05_perf
  )
}

# ---- reference predictions for the Python ports -----------------------------

# The v05 fit plus two off-boundary settings, so decay (C < 1) and a weak
# uncertainty bias are exercised too, not just kachergis's fitted C = 1.
kach_fit <- unlist(incumbents$kachergis$params)
REFERENCE_RUNS <- list(
  list(model = "baseline", constructor = "baseline", params = list()),
  list(model = "kachergis", constructor = "uncfam", fixed = list(variant = "entropy"),
       params = as.list(kach_fit)),
  list(model = "kachergis", constructor = "uncfam", fixed = list(variant = "entropy"),
       params = list(X = 0.5, B = 2, C = 0.9)),
  list(model = "kachergis", constructor = "uncfam", fixed = list(variant = "entropy"),
       params = list(X = 0.05, B = 0.5, C = 0.97))
)

run_reference <- function(run, datasets) {
  mod <- build_model(run, run$params)
  res <- xsl_run(mod, datasets)
  lapply(res$fits, \(f) list(
    perf = unname(as.numeric(f$perf)),
    matrix = unname(f$matrix),
    matrix_rows = rownames(f$matrix),
    matrix_cols = colnames(f$matrix),
    sse = ub(f$sse)
  ))
}

reference <- lapply(REFERENCE_RUNS, \(run) list(
  model = ub(run$model),
  constructor = ub(run$constructor),
  params = obj(run$params),
  fixed = obj(run$fixed),
  corpus = run_reference(run, xsl_datasets)
))

# ---- write ------------------------------------------------------------------

outputs <- c(
  write_json(corpus, "corpus.json"),
  write_json(holdout, "holdout.json"),
  write_json(incumbents, "incumbents.json"),
  write_json(reference, "reference_predictions.json")
)

md5 <- \(p) unname(tools::md5sum(p))
manifest <- list(
  created = ub(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  xslmodels_commit = ub(xslmodels_commit),
  xslmodels_version = ub(as.character(packageVersion("XSLmodels", lib.loc = pinned_lib))),
  r_version = ub(R.version.string),
  fits = list(path = ub("xsl-bakeoff/fits/v05/adults/group_fits.Rdata"), md5 = ub(md5(fits_path)),
              updated = ub(fits_env$meta$updated)),
  n_conditions = ub(length(corpus)),
  holdout_sets = scorable,
  incumbents_skipped = skipped,
  files = setNames(lapply(outputs, \(p) ub(md5(p))), basename(outputs))
)
write_json(manifest, "manifest.json")
cat("Wrote", length(outputs) + 1, "files to", out_dir, "\n")
if (length(skipped)) cat("No v05 group fit yet for:", paste(skipped, collapse = ", "), "\n")
