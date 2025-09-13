/// <reference types="react" />

declare module "*.png" {
  const value: string;
  export default value;
}
declare module "*.css" {
  const value: string;
  export default value;
}
declare module "*.jpg" {
  const value: string;
  export default value;
}
declare module "*.jpeg" {
  const value: string;
  export default value;
}
declare module "*.gif" {
  const value: string;
  export default value;
}
declare module "*.webp" {
  const value: string;
  export default value;
}
declare module "*.avif" {
  const value: string;
  export default value;
}
declare module "*.svg" {
  import * as React from "react";
  // If you import svg as URL: import logo from './logo.svg'
  const src: string;
  export default src;
  // If you use SVGR's ReactComponent: import { ReactComponent as Logo } from './logo.svg'
  export const ReactComponent: React.FunctionComponent<
    React.SVGProps<SVGSVGElement> & { title?: string }
  >;
}
