import React, { Fragment } from "react";

interface TabProps {
  title: string;
  content?: string;
}

const TabAutomatic = ({ title, content }: TabProps) => {
  return <Fragment>
    <h3>{title}</h3>
    <p>{content}</p>
  </Fragment>;
};

export default TabAutomatic;
