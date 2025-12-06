import { Box, Button, CircularProgress } from "@mui/material";
import { GridOverlay } from "@mui/x-data-grid";
import React from "react";

const CustomLoadingOverlay = () => {
  return (
    <GridOverlay>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          width: "100%",
          gap: 1,
        }}
      >
        <CircularProgress size={48} />
        <div className="text-lg text-primary">Carregando dados...</div>
      </Box>
    </GridOverlay>
  );
};

// Custom "no rows" overlay. Accepts an optional message via props
type NoRowsOverlayProps = {
  message?: string;
  componentType?: "products" | "businessDetails";
};

export function CustomNoRowsOverlay(props: NoRowsOverlayProps) {
  const { message = "Nenhum dado encontrado." } = props;

  return (
    <GridOverlay>
      <Box
        sx={{
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 1,
          p: 2,
        }}
      >
        <div className="text-lg font-medium text-primary">{message}</div>
        {props.componentType === "products" && (
          <div>
            <div className="text-sm text-gray-300">
              Se quiser, adicione produtos manualmente ou importe um arquivo.
            </div>
            <Box sx={{ mt: 2 }}>
              {/* Example call to "open add product" - replace with real handler */}
              <Button variant="outlined" color="secondary" size="small">
                Adicionar produto
              </Button>
            </Box>
          </div>
        )}
      </Box>
    </GridOverlay>
  );
}

export default CustomLoadingOverlay;
