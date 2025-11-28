import { QuartzComponentConstructor } from "./types"

function HorizontalDivider() {
  return <hr class="horizontal-divider" />
}

export default (() => HorizontalDivider) satisfies QuartzComponentConstructor
