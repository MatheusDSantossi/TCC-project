import type { ReactNode } from "react";

/**
 * A leaf item in the nav: it represents a concrete "route/component"
 * - `component` is required for leaves.
 * - `icon` and `handler` are optional.
 */
export type NavLeaf = {
  icon?: ReactNode;
  component: string;
  handler?: (e: React.MouseEvent, ...args: any[]) => void;
};

/**
 * A group can contain subitems (which are leaves). Groups may optionally
 * have a `component` too (treat as a index page), but it's not required.
 */

export type NavGroup = {
  icon?: ReactNode;
  subitems: Record<string, NavLeaf | NavGroup>;
  component?: string;
  handler?: (e: React.MouseEvent, ...args: any[]) => void;
};

// A nav item is either a group (has subitems) or a leaf (component present).
export type NavItem = NavLeaf | NavGroup;

export type NavTree = Record<string, NavItem>;
