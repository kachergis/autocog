# Reference predictions of the corrected tilles() for the Python port.
#
# XSLmodels after the e70c10c pin corrects tilles() (1b407e7, d07afff,
# 3f10da2, 7ad8432: eq 1's 1/C, eq 2's denominator, eq 7's entropy
# normalised by log2(n_o) for every seen word, and even flux on zero
# denominators). xsl-bakeoff installed that version in fits/v05/tilles/Rlib
# and refit tilles with it. This writes data/xsl/reference_tilles.json: the
# refit's parameters and SSE, and xsl_run()'s perf and final matrix on every
# corpus condition at the refit and at two other settings (one with a low
# alpha_0, which drove associations negative under the old eq 7). It also
# writes `incumbent`, the incumbents.json entry for the refit (with v05's own
# per-item ModelPerf, and the library it runs in), which
# scripts/xsl/update_tilles_incumbent.py copies into incumbents.json.
#
#   Rscript scripts/xsl/export_tilles_reference.R [--bakeoff ../xsl-bakeoff] [--out data/xsl]

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[[i + 1]]
}
bakeoff <- normalizePath(arg_value("--bakeoff", "../xsl-bakeoff"), mustWork = TRUE)
out_dir <- arg_value("--out", "data/xsl")

lib <- file.path(bakeoff, "fits", "v05", "tilles", "Rlib")
stopifnot(dir.exists(file.path(lib, "XSLmodels")))
.libPaths(c(lib, .libPaths()))
suppressMessages(library(XSLmodels, lib.loc = lib))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))
commit <- trimws(readLines(file.path(bakeoff, "fits", "v05", "tilles", "XSLMODELS_COMMIT")))

fits_path <- file.path(bakeoff, "fits", "v05", "adults", "group_fits.Rdata")
fits_env <- new.env()
load(fits_path, envir = fits_env)
fit <- fits_env$group_fits$tilles
refit <- setNames(as.list(unname(fit$optim$bestmem)), c("x", "b", "alpha_0"))

ub <- jsonlite::unbox
run_reference <- function(params) {
  res <- xsl_run(do.call(tilles, params), xsl_datasets)
  list(
    params = lapply(params, ub),
    sse = ub(res$sse),
    corpus = lapply(res$fits, \(f) list(
      perf = unname(as.numeric(f$perf)),
      matrix = unname(f$matrix),
      matrix_rows = rownames(f$matrix),
      matrix_cols = colnames(f$matrix)
    ))
  )
}

rows <- fits_env$gfd[fits_env$gfd$Model == "tilles", ]
v05_perf <- lapply(xsl_datasets, \(d) as.numeric(rows$ModelPerf[rows$Condition == d$condition &
                                                                   rows$condnum == d$label]))
incumbent <- list(
  name = ub("tilles"), constructor = ub("tilles"), params = lapply(refit, ub),
  fixed = setNames(list(), character(0)), stochastic = ub(FALSE), sse_v05 = ub(fit$optim$bestval),
  n_params = ub(3L), v05_perf = v05_perf, rlib = ub("tilles/Rlib"), xslmodels_commit = ub(commit)
)

out <- list(
  xslmodels_commit = ub(commit),
  xslmodels_version = ub(as.character(packageVersion("XSLmodels", lib.loc = lib))),
  group_fits_md5 = ub(unname(tools::md5sum(fits_path))),
  refit = list(params = lapply(refit, ub), sse = ub(fit$optim$bestval)),
  incumbent = incumbent,
  runs = list(
    run_reference(refit),
    run_reference(list(x = 0.6, b = 0.8, alpha_0 = 0.85)),
    run_reference(list(x = 0.3, b = 0.1, alpha_0 = 0.2))
  )
)
path <- file.path(out_dir, "reference_tilles.json")
jsonlite::write_json(out, path, digits = NA)
cat("wrote", path, "at XSLmodels", commit, "\n")
