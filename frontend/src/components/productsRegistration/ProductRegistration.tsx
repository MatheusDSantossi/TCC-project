import { useState } from "react";
import ProductRegistrationByFile from "./ProductRegistrationByFile";
import TabManual from "./tabs/TabManual";
import Tabs from "./tabs/Tabs";

type TabsType = {
  label: string;
  index: number;
  Component: React.FC<{}>;
}[];

// Tabs Array
const tabs: TabsType = [
  {
    label: "Cadastro Manual",
    index: 0,
    Component: TabManual,
  },
  {
    label: "Cadastro Automático",
    index: 1,
    Component: ProductRegistrationByFile,
  },
];

function ProductRegistration() {
  const [selectedTab, setSelectedTab] = useState<number>(tabs[0].index);

  return (
    <div className="flex justify-center items-center bg-tertiary h-screen">
      <div className="flex flex-col items-start justify-start bg-tertiary">
        <div className="w-full flex justify-start items-start">
          <Tabs
            selectedTab={selectedTab}
            onClick={setSelectedTab}
            tabs={tabs}
          />
        </div>
      </div>
    </div>
  );
}

export default ProductRegistration;
