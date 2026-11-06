/** Join class names, skipping falsy ones (the `class:` directive of the Svelte originals). */
export const cx = (...names: (string | false | null | undefined)[]) => names.filter(Boolean).join(" ");
