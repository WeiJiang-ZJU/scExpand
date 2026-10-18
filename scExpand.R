# scExpand: stratified pseudo-sampling for CyTOF, scRNA-seq, and related data.
# Source and citation: https://github.com/WeiJiang-ZJU/scExpand
# Original implementation: R 4.4.1. See README.md for the algorithm and usage.

#' Expand a feature-by-cell matrix into sample-level pseudo-samples.
#'
#' @param expression Numeric matrix (proteins or genes x cells), optionally sparse.
#' @param metadata Data frame with cell IDs as row names and columns sample,
#'   cluster, Patient, Outcome, and optional Time (defaults to 'single').
#' @param k Fraction sampled without replacement within each observed cluster.
#' @param A Number of pseudo-samples per original sample.
#' @param seed Integer seed, or NULL to use the current random-number stream.
#' @param clusters Ordered cluster universe; absent sample-cluster pairs are zero.
#' @return Data frame with cluster__marker features and source metadata.
expand_matrix <- function(expression, metadata, k = 0.2, A = 100L,
                          seed = 24L, clusters = NULL) {
  if (length(k) != 1L || !is.numeric(k) || !is.finite(k) || k <= 0 || k > 1)
    stop("k must be a fraction in (0, 1].")
  if (length(A) != 1L || !is.numeric(A) || !is.finite(A) ||
      A < 1 || A != floor(A)) stop("A must be a positive integer.")
  if (length(dim(expression)) != 2L || any(dim(expression) == 0L))
    stop("expression must be a nonempty marker-by-cell matrix.")
  values <- if (inherits(expression, "sparseMatrix")) expression@x else expression
  if (!is.numeric(values) || any(!is.finite(values)))
    stop("Expression values must be numeric and finite.")
  markers <- rownames(expression)
  cells <- colnames(expression)
  valid_ids <- function(x) !is.null(x) && !anyNA(x) &&
    all(nzchar(x)) && !anyDuplicated(x)
  if (!valid_ids(markers) || !valid_ids(cells))
    stop("Expression row and column names must be nonempty and unique.")
  required <- c("sample", "cluster", "Patient", "Outcome")
  if (!is.data.frame(metadata) || !all(required %in% names(metadata)))
    stop("metadata must contain sample, cluster, Patient, and Outcome.")
  if (!"Time" %in% names(metadata)) metadata$Time <- "single"
  required <- c(required, "Time")
  if (!valid_ids(rownames(metadata)) || !all(cells %in% rownames(metadata)))
    stop("metadata row names must uniquely identify all expression cells.")
  metadata <- metadata[cells, required, drop = FALSE]
  metadata[] <- lapply(metadata, as.character)
  if (anyNA(metadata) || any(!nzchar(as.matrix(metadata))))
    stop("Required metadata must not contain missing or empty values.")
  if (is.null(clusters)) clusters <- sort(unique(metadata$cluster))
  clusters <- as.character(clusters)
  if (!valid_ids(clusters) || !all(metadata$cluster %in% clusters))
    stop("clusters must be unique and contain all observed cluster labels.")
  if (any(grepl("__", c(clusters, markers), fixed = TRUE)))
    stop("Cluster and marker names must not contain the reserved separator '__'.")
  feature_names <- unlist(lapply(clusters, function(cl)
    paste(cl, markers, sep = "__")), use.names = FALSE)
  if (!is.null(seed)) {
    if (length(seed) != 1L || !is.numeric(seed) || !is.finite(seed) ||
        seed < 0 || seed > .Machine$integer.max || seed != floor(seed))
      stop("seed must be a nonnegative integer or NULL.")
    had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
    on.exit({
      if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
      else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
        rm(".Random.seed", envir = .GlobalEnv)
    }, add = TRUE)
    set.seed(seed)
  }
  sample_ids <- unique(metadata$sample)
  result <- vector("list", length(sample_ids))
  for (s in seq_along(sample_ids)) {
    sample_id <- sample_ids[[s]]
    idx <- which(metadata$sample == sample_id)
    info <- lapply(metadata[idx, c("Patient", "Outcome", "Time"), drop = FALSE], unique)
    if (any(lengths(info) != 1L))
      stop("Each sample must have exactly one Patient, Outcome, and Time.")
    block <- matrix(0, nrow = A, ncol = length(feature_names),
                    dimnames = list(NULL, feature_names))
    pools <- lapply(clusters, function(cl) idx[metadata$cluster[idx] == cl])
    for (i in seq_len(A)) {
      for (j in seq_along(clusters)) {
        pool <- pools[[j]]
        if (!length(pool)) next
        n <- max(1L, floor(length(pool) * k))
        # sample.int avoids R's sample(x) special case when x is one integer.
        selected <- pool[sample.int(length(pool), size = n, replace = FALSE)]
        cols <- (j - 1L) * length(markers) + seq_along(markers)
        block[i, cols] <- rowMeans(as.matrix(expression[, selected, drop = FALSE]))
      }
    }
    df <- as.data.frame(block, check.names = FALSE)
    df$Outcome <- info$Outcome
    df$Patient <- info$Patient
    df$Time <- info$Time
    df$sample <- sample_id
    df$pseudo_sample_id <- paste0("sample", s, "_pseudo", seq_len(A))
    result[[s]] <- df
  }
  result <- do.call(rbind, result)
  rownames(result) <- NULL
  attr(result, "feature_manifest") <- data.frame(
    feature = feature_names,
    cluster = rep(clusters, each = length(markers)),
    marker = rep(markers, times = length(clusters)),
    stringsAsFactors = FALSE
  )
  result
}

#' Expand a Seurat object using a selected expression assay and layer.
#' For scRNA-seq, use assay = 'RNA' and layer = 'data' or 'scale.data'.
#' Requires SeuratObject; preprocessing and cluster annotation are external.
expand_samples <- function(seurat_obj, k = 0.2, A = 100L, seed = 24L,
                           assay = "integrated", layer = "scale.data",
                           markers = NULL, clusters = NULL) {
  if (!requireNamespace("SeuratObject", quietly = TRUE))
    stop("Install SeuratObject to use expand_samples().")
  if (!inherits(seurat_obj, "Seurat")) stop("seurat_obj must be a Seurat object.")
  expression <- if (utils::packageVersion("SeuratObject") >= "5.0.0") {
    SeuratObject::GetAssayData(seurat_obj, assay = assay, layer = layer)
  } else {
    SeuratObject::GetAssayData(seurat_obj, assay = assay, slot = layer)
  }
  if (!is.null(markers)) {
    if (!length(markers) || anyNA(markers) || anyDuplicated(markers) ||
        !all(markers %in% rownames(expression)))
      stop("markers must be unique names present in the requested assay layer.")
    expression <- expression[markers, , drop = FALSE]
  }
  expand_matrix(expression, seurat_obj[[]], k, A, seed, clusters)
}
