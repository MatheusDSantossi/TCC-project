import { useState } from "react";
import logo from "../assets/logo.svg";
import Papa from "papaparse";
import AxiosInstance from "./AxiosInstance";
import { useException } from "../context/ExceptionContext";
import { ExceptionVariant } from "../exceptions/ExceptionVariantsTypes";

type Row = {
  name?: string;
  brand?: string;
  quantity?: string;
  section?: string | number;
  price?: string | number;
  unit?: string;
  stock?: string | number;
  synonymous?: string;
};

// TODO: I need to create a better function to bigger files, and this new alternative will need backend handler. So, first let's test this one

function normalizeRow(row: Row) {
  // Map/cleanup column names used in your CSV/XLSX
  // Accept flexible column names: name, nome, productName etc.
  const r = {
    name: (row.name || (row as any).nome || "").toString().trim(),
    brand: (row.brand || (row as any).marca || "").toString().trim(),
    quantity: (row.quantity || (row as any).quantidade || "").toString().trim(),
    section: Number(row.section ?? (row as any).secao ?? 0),
    price: Number(row.price ?? (row as any).preco ?? 0),
    unit: (row.unit || (row as any).unidade || "").toString().trim(),
    stock: Number(row.stock ?? (row as any).estoque ?? 0),
    // synonyms: try JSON array, else split by | or ;
    synonymous: (() => {
      const s = row.synonymous ?? (row as any).sinonimos ?? "";
      if (!s) return [];
      try {
        const parsed = JSON.parse(s.toString());
        if (Array.isArray(parsed)) return parsed;
      } catch {}
      return s
        .toString()
        .split(/[|;]+/)
        .map((x) => x.trim())
        .filter(Boolean);
    })(),
  };
  return r;
}

const ProductRegistrationByFile = () => {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  const { showException } = useException();

  const parseCsv = (f: File): Promise<Row[]> =>
    new Promise((res, rej) => {
      Papa.parse<Row>(f, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => res(results.data),
        error: (err) => rej(err),
      });
    });

  const handleFile = (f: File | null) => setFile(f);

  const uploadParsed = async (rows: Row[]) => {
    if (rows.length === 0) return;

    const batchSize = 200; // tune based on backend limits
    setProgress({ done: 0, total: rows.length });

    for (let i = 0; i < rows.length; i += batchSize) {
      const batch = rows.slice(i, i + batchSize).map(normalizeRow);
      // POST each row to the single-product endpoint
      // We can also implement here a bulk endpoint for efficiency
      const promises = batch.map((r) =>
        AxiosInstance.post("/api/products/", {
          name: r.name,
          brand: r.brand,
          quantity: r.quantity,
          section: r.section,
          price: r.price,
          unit: r.unit,
          stock: r.stock,
          synonymous: r.synonymous,
        })
      );
      //   Wait for the batch
      await Promise.allSettled(promises);
      setProgress((p) => ({
        ...p,
        done: Math.min(p.total, p.done + batch.length),
      }));
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    try {
      let rows: Row[] = [];
      if (file.name.endsWith(".csv")) {
        rows = await parseCsv(file);
      } else {
        // rows = await Parsexls
      }

      await uploadParsed(rows);
      showException({
        text: "Upload completo!",
        // error: error,
        variant: ExceptionVariant.INFO,
      });
    } catch (err) {
      showException({
        text: "Error parsing or uploading file",
        error: err,
        variant: ExceptionVariant.ERROR,
      });
    }
  };

  return (
    <div className="justify-center">
      <img className="w-20 mr-4" src={logo} alt="Hecate's logo" />
      <h2 className="mb-6 text-2xl font-semibold text-center left-40">
        Cadastro usando arquivo do produto
      </h2>
      <input
        className="block shadow-2xl! w-full text-sm
              file:mr-4 file:py-2 file:px-3
              file:border-0
              file:text-sm file:font-semibold
              file:bg-violet-50 file:text-primary
              hover:file:bg-secondary-dark
              file:cursor-pointer
              cursor-pointer
              border border-white hover:border-secondary-dark rounded-lg
              focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent
              file:transition file:duration-300 file:ease-in-out"
        type="file"
        accept=".xlsx, .csv"
        id="file_select"
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />
      <button onClick={handleUpload} disabled={!file}>
        Upload
      </button>
      <div>
        Uploaded {progress.done} / {progress.total}
      </div>
    </div>
  );
};

export default ProductRegistrationByFile;
