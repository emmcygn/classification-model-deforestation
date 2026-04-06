import { useCallback, useState } from "react";

interface Props {
  targetId: string;
  filename?: string;
}

export default function ExportButton({ targetId, filename = "deforestai-brief" }: Props) {
  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async () => {
    const element = document.getElementById(targetId);
    if (!element) {
      alert(`Export target not found. Make sure the report is fully loaded before exporting.`);
      return;
    }

    setExporting(true);
    try {
      const html2canvas = (await import("html2canvas")).default;
      const { jsPDF } = await import("jspdf");

      // Clone element to avoid scroll/visibility issues
      const clone = element.cloneNode(true) as HTMLElement;
      clone.style.position = "absolute";
      clone.style.left = "-9999px";
      clone.style.top = "0";
      clone.style.width = `${element.offsetWidth}px`;
      clone.style.background = "#1f2937";
      clone.style.padding = "24px";
      clone.style.color = "#e5e7eb";
      document.body.appendChild(clone);

      // Add metadata header to clone
      const header = document.createElement("div");
      header.style.cssText = "padding: 12px 0 16px 0; border-bottom: 1px solid #374151; margin-bottom: 16px;";
      header.innerHTML = `
        <div style="font-size: 18px; font-weight: bold; color: #e5e7eb;">DeforestAI Report</div>
        <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">
          Generated: ${new Date().toLocaleString()} | deforestai-philippines
        </div>
      `;
      clone.insertBefore(header, clone.firstChild);

      const canvas = await html2canvas(clone, {
        backgroundColor: "#1f2937",
        scale: 2,
        useCORS: true,
        logging: false,
      });

      document.body.removeChild(clone);

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({ orientation: "p", unit: "mm", format: "a4" });
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgHeight = (canvas.height * pdfWidth) / canvas.width;

      let heightLeft = imgHeight;
      let position = 0;

      pdf.addImage(imgData, "PNG", 0, position, pdfWidth, imgHeight);
      heightLeft -= pdfHeight;

      while (heightLeft > 0) {
        position = -(imgHeight - heightLeft);
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 0, position, pdfWidth, imgHeight);
        heightLeft -= pdfHeight;
      }

      pdf.save(`${filename}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
      alert("PDF export failed. Try again or use browser Print (Ctrl+P).");
    } finally {
      setExporting(false);
    }
  }, [targetId, filename]);

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      className="w-full px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 rounded text-xs font-medium transition-colors"
    >
      {exporting ? "Exporting..." : "Export as PDF"}
    </button>
  );
}
