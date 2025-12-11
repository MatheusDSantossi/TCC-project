import { useRef, useState } from "react";
import logo from "../../assets/logo.svg";
import Papa from "papaparse";
import AxiosInstance from "../AxiosInstance";
import { useException } from "../../context/ExceptionContext";
import { ExceptionVariant } from "../../exceptions/ExceptionVariantsTypes";
import CustomButton from "../forms/Button";
import ReactiveButton from "reactive-button";
import { progress } from "framer-motion";
import type { AxiosProgressEvent } from "axios";
import { LinearProgress } from "@mui/material";
import Spinner from "../../ui/Spinner";

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
  // States
  const [file, setFile] = useState<File | null>(null);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [processingPercent, setProcessingPercent] = useState<number | null>(
    null
  );
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [buttonState, setButtonState] = useState("idle");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isProcessing, setIsProcessing] = useState(false);

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

  // * IMPORTANT: To test out this feature I need first to create a new business, so it won't affect the businesses I have now.

  const uploadParsed = async (rows: Row[]) => {
    if (rows.length === 0) return;

    const batchSize = 200; // tune based on backend limits
    setProgress({ done: 0, total: rows.length });

    console.log("rows: ", rows);

    for (let i = 0; i < rows.length; i += batchSize) {
      const batch = rows.slice(i, i + batchSize).map(normalizeRow);
      // POST each row to the single-product endpoint
      // We can also implement here a bulk endpoint for efficiency
      const promises = batch.map((r) =>
        AxiosInstance.post("products/", {
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

  const clearFileSelection = () => {
    // Clear React state
    setFile(null);

    // Clear the native file input so filename disappears
    if (fileInputRef.current) {
      try {
        fileInputRef.current.value = "";
      } catch (e) {
        // some browsers may be picky; we ignore errors here
        console.warn("Could not clear file input programmatically", e);
      }
    }
  };

  const uploadFile = async (file: File) => {
    if (!file) return;

    const form = new FormData();
    form.append("file", file, file.name);
    setButtonState("loading");
    setIsProcessing(true);

    console.log("file: ", file);
    form.append("file", file);

    setUploadPercent(0);
    setProcessingPercent(null);
    setStatusMessage("Iniciando upload...");

    try {
      const res = await AxiosInstance.post("products/import/", form, {
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          const percent = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
        },
      });
      // Server could respond:
      // { job_id: "abcd-1234" }  -> you should poll job status endpoint
      // or { created: 100, errors: [...] } -> synchronous result
      // if (res.data?.job_id) {
      setButtonState("success");
      setStatusMessage("Upload completo! Processando dados...");
      // } else {
      //   setButtonState("error");
      //   setStatusMessage("Falha ao importar os produtos");
      //   showException({
      //     text: "Falha ao importar os produtos",
      //     error: res?.data ?? null,
      //     variant: ExceptionVariant.ERROR,
      //   });
      // }

      console.log("response: ", res?.data);
    } catch (err) {
      setButtonState("error");
      console.error("Error uploading file:", err);
      setStatusMessage("Erro ao importar os produtos");
      showException({
        text: "Falha ao importar os produtos",
        error: err?.response?.data ?? err,
        variant: ExceptionVariant.ERROR,
      });
    } finally {
      // Always clear file selection (so UI is safe for next upload)
      clearFileSelection();
      setUploadPercent(null);
      setIsProcessing(false);
      setTimeout(() => setButtonState("idle"), 1500);
    }
  };

  // Polling helper for background job status
  async function pollJobStatus(jobId: string) {
    const intervalMs = 1500;
    setProcessingPercent(0);

    // polling loop
    return new Promise<void>((resolve) => {
      const timer = setInterval(async () => {
        try {
          const r = await AxiosInstance.get(
            `products/import/status/${jobId}/`
          );
          // Example server response: { status: 'PENDING'|'PROGRESS'|'SUCCESS'|'FAILURE', progress: 45, message: 'parsing' }
          const data = r.data;

          if (data.progress != null) setProcessingPercent(data.progress);
          if (data.message) setStatusMessage(data.message);

          if (data.status === "SUCCESS" || data.status === "FAILURE") {
            clearInterval(timer);

            setStatusMessage(
              data.status === "SUCCESS"
                ? "Importação completa!"
                : `Falha na importação dos produtos. ${data.error ?? ""}`
            );
            resolve();
          }
        } catch (err) {
          console.warn("Status poll error ", err);
        }
      }, intervalMs);
    });
  }

  return (
    <div className="w-full md:w-[570px] xl:w-4xl md:h-[660px] xl:h-[630px] max-w-4xl p-8 rounded-lg shadow-xl bg-primary  transition-all duration-300 ease">
      <Spinner open={isProcessing} text="Importando produtos - aguarde..." />
      <div className="justify-center">
        <div className="flex items-center mb-7">
          <img className="w-20 mr-4" src={logo} alt="Hecate's logo" />
          <h2 className="mb-6 m-auto text-2xl font-semibold text-center left-40">
            Cadastro usando arquivo do produto
          </h2>
        </div>
        <input
          ref={fileInputRef}
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
          accept=".xlsx, .csv, .xls"
          id="file_select"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {file && (
          <div className="mt-2 text-sm text-white">
            Arquivo selecionado: "{file.name}"
            <button
              type="button"
              onClick={clearFileSelection}
              className="ml-3 text-xs text-red-600"
            >
              Remover
            </button>
          </div>
        )}
        <div className="mt-5">
          <ReactiveButton
            className="text-primary hover:bg-yellow-300"
            disabled={!file}
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Upload feito com sucesso"}
            idleText={"Upload"}
            animation={true}
            rounded={true}
            width={"120"}
            height={"40"}
            size="large"
            // onClick={handleUpload}
            onClick={() => {
              uploadFile(file);
            }}
          />
        </div>

        {uploadPercent != null && (
          <div>
            <div>Uploading: {uploadPercent}%</div>
            <LinearProgress variant="determinate" value={uploadPercent} />
          </div>
        )}

        {processingPercent != null && (
          <div>
            <div>Processing: {processingPercent}%</div>
            <LinearProgress variant="determinate" value={processingPercent} />
          </div>
        )}

        {statusMessage && <div>{statusMessage}</div>}
        {/* <div>
          Uploaded {progress.done} / {progress.total}
        </div> */}
      </div>
    </div>
  );
};

export default ProductRegistrationByFile;
