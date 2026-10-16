# Export per-subject test responses for every xsl_datasets condition to
# data/xsl/subject_responses.json, for the random-ability likelihood
# (src/xsl/fitting.py, objective "subject").
#
# The group-level corpus (corpus.json) only has per-item counts, which cannot
# identify between-subject differences: each person answers each item once,
# so an item's count is binomial however much people differ. Per-subject
# response patterns can. XSLmodels keeps them in data-raw/; this reuses its own
# loader (data-raw/raw_responses.R) and condition matching
# (data-raw/extract_response_matrices.R) without writing anything into the
# XSLmodels checkout, then pivots each condition's responses into a
# subject x item 0/1 matrix in corpus item order.
#
# Refuses to write unless every matrix reproduces the condition's stored
# per-item accuracy exactly and has n_subj rows.
#
# Run from the autocog repo root, after scripts/xsl/export_corpus.R (which
# rewrites manifest.json), with the XSLmodels checkout at the commit in
# data/xsl/manifest.json:
#   Rscript scripts/xsl/export_subject_responses.R [--xslmodels ../XSLmodels] [--out data/xsl]

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[[i + 1]]
}
xslmodels <- normalizePath(arg_value("--xslmodels", "../XSLmodels"), mustWork = TRUE)
out_dir <- normalizePath(arg_value("--out", "data/xsl"), mustWork = TRUE)

manifest <- jsonlite::read_json(file.path(out_dir, "manifest.json"))
head_commit <- system2("git", c("-C", xslmodels, "rev-parse", "HEAD"), stdout = TRUE)
if (!identical(head_commit, manifest$xslmodels_commit)) {
  stop("XSLmodels checkout is at ", head_commit, " but data/xsl was exported from ",
       manifest$xslmodels_commit, "; check out that commit first")
}

old_wd <- setwd(xslmodels)
suppressMessages(devtools::load_all(".", quiet = TRUE))
saveRDS <- function(...) invisible(NULL)  # extract_response_matrices.R saves a review file; don't
invisible(capture.output(suppressWarnings({
  source("data-raw/raw_responses.R")
  source("data-raw/extract_response_matrices.R", local = TRUE)
})))
setwd(old_wd)
suppressMessages(library(dplyr))

stopifnot(all(report$status == "exact"))

export_one <- function(d) {
  row <- report[report$label == d$label, ]
  groups <- strsplit(row$groups, " \\+ ")[[1]]
  rows <- lapply(seq_along(groups), function(k) {
    g <- raw_groups[[groups[k]]]
    # filtering conditions: items 1-12 are the AtoA tests, 13-24 the AtoB tests
    offset <- if (grepl("^filt", d$label) && grepl("AtoB", groups[k])) 12L else 0L
    tibble(subject = g$subject, item = g$word + offset, correct = g$correct)
  }) |> bind_rows()
  n_items <- length(d$accuracy)
  stopifnot(all(rows$item >= 1 & rows$item <= n_items))
  dup <- rows |> count(subject, item) |> filter(n > 1)
  if (nrow(dup)) stop(d$label, ": a subject answered an item twice")
  subjects <- sort(unique(rows$subject))
  m <- matrix(NA_integer_, length(subjects), n_items)
  m[cbind(match(rows$subject, subjects), rows$item)] <- as.integer(rows$correct)
  acc <- colMeans(m, na.rm = TRUE)
  if (length(subjects) != d$n_subj) stop(d$label, ": ", length(subjects), " subjects, n_subj = ", d$n_subj)
  if (max(abs(acc - d$accuracy)) > 1e-9) stop(d$label, ": per-item accuracy does not reproduce the corpus")
  list(label = jsonlite::unbox(as.character(d$label)), subjects = subjects,
       n_missing = jsonlite::unbox(sum(is.na(m))), correct = unname(m))
}

out <- lapply(xsl_datasets, export_one)
path <- file.path(out_dir, "subject_responses.json")
jsonlite::write_json(out, path, auto_unbox = FALSE, na = "null", pretty = FALSE)

manifest$files[["subject_responses.json"]] <- jsonlite::unbox(unname(tools::md5sum(path)))
manifest$subject_responses <- list(source = jsonlite::unbox("XSLmodels data-raw/raw_responses.R"),
                                   n_conditions = jsonlite::unbox(length(out)),
                                   n_subjects = jsonlite::unbox(sum(vapply(out, \(x) length(x$subjects), 0))))
jsonlite::write_json(manifest, file.path(out_dir, "manifest.json"), auto_unbox = TRUE, digits = NA, pretty = FALSE)
cat("Wrote", path, ":", length(out), "conditions,",
    sum(vapply(out, \(x) length(x$subjects), 0)), "subjects,",
    sum(vapply(out, \(x) x$n_missing, 0)), "missing responses\n")
