# Publish a release

The `Publish Python Package` workflow builds a wheel and source distribution,
runs unit tests for the release commit, and installs the wheel in a separate
integration job. That job generates TX-3-zone inputs from a pinned
PowerGenome-examples revision and solves the first planning period with GenX
0.4.5 and HiGHS. Publishing requires both test jobs to succeed, including an
`OPTIMAL` GenX termination status.

The distributions are built once and passed between jobs as an artifact.
Build and test jobs have read-only repository permissions, without publishing
credentials. The publish job uses OIDC trusted publishing and a GitHub
environment; no PyPI API token is needed.

## One-time configuration

1. Merge the release workflow into `main`. The manual **Run workflow** button
   becomes available only after the workflow exists on the default branch.
2. In **Settings > Actions > General**, allow the actions used by the workflow.
3. In **Settings > Environments**, create or configure `release`. Consider
   requiring a maintainer's approval. If restricting deployment branches and
   tags, allow both `main` (manual runs) and `v*` tags (automatic runs).
   Environment restrictions apply to the workflow's ref, not the tag input
   checked out during a manual run.
4. On PyPI, register a trusted publisher for `PowerGenome/PowerGenome`, workflow
   filename `release.yml`, environment `release`.

Consider using a tag ruleset to restrict creation, updates, and deletion of
release tags to maintainers. Do not move an existing published version tag.

## Future releases

1. Update `powergenome/version.py` and merge the change into `main`.
2. Tag that tested commit with its matching version, then push the tag:

   ```bash
   git tag v0.11.0 <release-commit>
   git push origin v0.11.0
   ```

3. Monitor **Actions > Publish Python Package**. Approve the deployment if the
   `release` environment requires approval.

The workflow rejects tags that do not match the built package version.
After tests pass, it publishes the tested distributions to PyPI and creates
the GitHub release with generated notes and distribution attachments. If a
GitHub release already exists, its notes are preserved and files are attached
without overwriting existing assets. Publishing a GitHub release for an
already-existing tag is not itself a workflow trigger.

## Publish an existing tag

In **Actions > Publish Python Package > Run workflow**, select branch `main`
and enter an explicit existing tag, such as `v0.11.0`.
The wheel, source distribution, and unit tests come from that tag's commit,
not from `main`. The integration gate uses the pinned example and GenX version
in the workflow, so an incompatible historical release will fail the gate
rather than bypassing it.

PyPI does not permit overwriting uploaded distribution filenames.
This workflow deliberately does not skip existing files. Use a new version
for a corrected release; do not rerun an already-published version to replace
its files.

If publishing succeeds but creating the GitHub release fails, rerun only the
failed `github-release` job from the workflow run. It uses the same distribution
artifact, without attempting another PyPI upload. Artifacts are subject to the
repository's retention policy.

## Diagnose a failed integration gate

Download the `release-genx-smoke` artifact from the workflow run. It contains
the generated PowerGenome inputs, pipeline log, GenX log and results (when
available), and the resolved Julia project and manifest. The `release-distributions`
artifact contains the exact wheel and source distribution under test.

When updating the pinned example commit or GenX version in `release.yml`, run
the PR's release checks before merging. These checks exercise the build and
solve without publishing.
