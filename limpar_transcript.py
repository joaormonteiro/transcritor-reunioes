import re
import sys


def limpar(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    resultado = []
    frase_atual = ""

    for line in lines:
        line = line.strip()

        if not line or line.startswith('#'):
            if frase_atual:
                resultado.append(frase_atual)
                frase_atual = ""
            resultado.append(line)
            continue

        texto = re.sub(r'^\[\d{2}:\d{2}\]\s*', '', line).strip()
        if not texto:
            continue

        if frase_atual:
            if frase_atual[-1] in '.!?:;':
                resultado.append(frase_atual)
                frase_atual = texto
            else:
                frase_atual += ' ' + texto
        else:
            frase_atual = texto

    if frase_atual:
        resultado.append(frase_atual)

    out = md_path.replace('.md', '_clean.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(resultado))
    print(f"Salvo: {out}")


if __name__ == "__main__":
    limpar(sys.argv[1])
